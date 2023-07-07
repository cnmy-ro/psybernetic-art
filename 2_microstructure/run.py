"""
Microstructure
"""

__author__ = "Chinmay Rao"



import os
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import vgg16
from tqdm import tqdm


IMAGENET_MEAN = torch.tensor([[[0.485, 0.456, 0.406]]]).permute(2,1,0)
IMAGENET_STD = torch.tensor([[[0.229, 0.224, 0.225]]]).permute(2,1,0)
DISPLAY_SIZE = 720


class FramesWriter:

	def __init__(self):
		self.frame_counter = 0
		self.frame_skip = 1
		self.output_dir = "./output/frames/"
		os.makedirs(self.output_dir, exist_ok=True)

	def write(self, frame):
		self.frame_counter += 1
		if self.frame_counter % self.frame_skip == 0:
			filename = str(self.frame_counter).zfill(5)
			plt.imsave(f"{self.output_dir}/{filename}.png", frame)


class VGG16Backbone(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        net = vgg16(pretrained=True).eval().to('cuda')
        self.backbone = net.features
        self.relevant_layer_idxs = [15, 18, 20, 22, 25, 27, 29]

    def forward(self, x):
        features = []
        for layer_idx in range(30):
            x = self.backbone[layer_idx](x)
            if layer_idx in self.relevant_layer_idxs:
                features.append(x)
        return features


def update(image, backbone, optimizer):

    optimizer.zero_grad()
    features = backbone(image)
    features = torch.cat([f.flatten() for f in features], dim=0)
    loss = -F.mse_loss(features, torch.zeros_like(features))
    loss.backward()
    optimizer.step()

    lower_image_bound = ((0 - IMAGENET_MEAN / IMAGENET_STD)).reshape(1, -1, 1, 1).to('cuda')
    upper_image_bound = ((1 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, -1, 1, 1).to('cuda')
    image.data = torch.max(torch.min(image, upper_image_bound), lower_image_bound)

    return image


def preprocess(image):
    dream_image = torch.tensor(image, dtype=torch.float)
    dream_image = torch.permute(dream_image, (2,0,1)).unsqueeze(0)
    dream_image = dream_image / 255.
    dream_image = (dream_image - IMAGENET_MEAN) / IMAGENET_STD
    return dream_image 


def postprocess(dream_image):
    dream_image = dream_image.detach().cpu()
    dream_image = dream_image * IMAGENET_STD + IMAGENET_MEAN
    dream_image = torch.clip(dream_image, 0.0, 1.0)
    image = dream_image.squeeze().permute(1,2,0).numpy()
    return image


def dream(image, writer):

    backbone = VGG16Backbone()
    dream_image = preprocess(image)
    image_hires = dream_image.clone().detach()
    orig_size = image_hires.shape[-1]

    zoom_factor = 1.02
    num_zooms = 420
    crop_size = orig_size

    # Initial still frames
    frame = postprocess(TF.resize(dream_image, DISPLAY_SIZE))
    for _ in range(60): writer.write(frame)

    # Run animation sequence
    for n in tqdm(range(num_zooms)):
        
        if n < 100:  # Simple zoom-in
            crop_size = round(crop_size / zoom_factor)
            dream_image = TF.center_crop(image_hires, (crop_size, crop_size))
            dream_image = TF.resize(dream_image, DISPLAY_SIZE).detach().to('cuda')

        elif n == 100:  # Inject some noise
            dream_image = dream_image + 0.25 * torch.randn_like(dream_image)

        else:  # Deep dream
            zoom_size = round(DISPLAY_SIZE * zoom_factor)
            dream_image = TF.resize(dream_image, zoom_size, interpolation=T.InterpolationMode.BICUBIC).detach()
            dream_image = TF.center_crop(dream_image, (DISPLAY_SIZE, DISPLAY_SIZE)).to('cuda')

            dream_image.requires_grad = True
            optimizer = torch.optim.Adam(params=[dream_image], lr=2e-3)
            
            for _ in range(15): dream_image = update(dream_image, backbone, optimizer)

        frame = postprocess(dream_image)
        for _ in range(2): writer.write(frame)

    # Final still frames
    for _ in range(45): writer.write(frame)


def frames_to_video():
    fps = 30
    video_writer = cv2.VideoWriter(f'./output/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (DISPLAY_SIZE, DISPLAY_SIZE))
    paths = sorted(glob.glob(f'./output/frames/*.png'))
    for i in range(len(paths)):
        frame = cv2.imread(paths[i])
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    image = plt.imread("./assets/foxglove.jpg")
    writer = FramesWriter()
    dream(image, writer)
    frames_to_video()