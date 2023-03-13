__author__ = "Chinmay Rao"


import os
import time
import random

import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, erosion, dilation, square, disk
from skimage import measure
from skimage.transform import resize
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import vgg16
from tqdm import tqdm



# ---
# Config

CONTOUR_PATHS = {
	'circle': "./assets/contours/vitruvian-circle.png", 'square': "./assets/contours/vitruvian-square.png", 
	'man-body1': "./assets/contours/vitruvian-man-body1.png", 'man-body2': "./assets/contours/vitruvian-man-body2.png", 
	'man-head1': "./assets/contours/vitruvian-man-head1.png", 'man-head2': "./assets/contours/vitruvian-man-head2.png",
	'man-detail-common': "./assets/contours/vitruvian-man-detail-common.png",
	'man-detail-body1': "./assets/contours/vitruvian-man-detail-body1.png",
	'man-detail-body2': "./assets/contours/vitruvian-man-detail-body2.png",
	}
MASK_PATHS = {
	'circle': "./assets/masks/vitruvian-circle.png", 
	'square': "./assets/masks/vitruvian-square.png",
	'man': "./assets/masks/vitruvian-man.png",
	'man-head': "./assets/masks/vitruvian-man-head.png"
	}
ART_PATHS = {
	'shade1': "./assets/art/vitruvian-man-shade1.png",
	'shade2': "./assets/art/vitruvian-man-shade2.png",
	'shade3': "./assets/art/vitruvian-man-shade3.png",
	'shade4': "./assets/art/vitruvian-man-shade4.png",
	'shade5': "./assets/art/vitruvian-man-shade5.png",
	'shade6': "./assets/art/vitruvian-man-shade6.png",
}
AUTOMATA_PATTERN_PATHS = {
	'p23honeyfarmhassler': "./assets/automata_patterns/p23honeyfarmhassler.txt",
	'twinbeesshuttle': "./assets/automata_patterns/twinbeesshuttle.txt"
}
PORTRAIT_PATHS = {
	'buddha': "./assets/portraits/buddha.jpg",
	'plato': "./assets/portraits/plato.jpg",
	'davinci': "./assets/portraits/davinci.png",
	'descartes': "./assets/portraits/descartes.jpg",
	'newton': "./assets/portraits/newton.jpg",
	'darwin': "./assets/portraits/darwin.jpg",
	'jung': "./assets/portraits/jung.jpg",
	'turing': "./assets/portraits/turing.jpg"
}

# Sizes in xy format; coords in upper-left xy format
IMAGE_SIZE = (691, 691)

SIGNAL_ORIGINS = {
	'circle': (IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2), 'square': (IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2), 
	'man-body1': (86, 521), 'man-body2': (605, 521), 'man-head1': (345, 33), 'man-head2': (345, 33)
	}
SIGNAL_MODS = {
	'circle': {'shift': (IMAGE_SIZE[0]//2, 33), 'reverse': True},
	'square': {'shift': (IMAGE_SIZE[0]//2, 143), 'reverse': False},
	'man-body1': {'shift': (303, 348), 'reverse': False}, 
	'man-body2': {'shift': (384, 348), 'reverse': True}, 
	'man-head1': {'shift': (IMAGE_SIZE[0]//2, 228), 'reverse': False},
	'man-head2': {'shift': None, 'reverse': True}
	}

# Colors in RGBA format
COLOR_WHITE = (1., 1., 1., 1.)
COLOR_BLACK = (0., 0., 0., 1.)
COLORS_STRUCT_ESTIM = {
	'circle': (0., 0.39, 0., 1.), 'square': (0.55, 0., 0., 1.),
	'man-body1': (0.863, 0.08, 0.24, 1.), 'man-body2': (0.13, 0.7, 0.67, 1.), 'man-head1': (0.58, 0., 0.83, 1.), 'man-head2': (0.58, 0., 0.83, 1.), 
	}
COLORS_HARMONIC_MAG = {
	'circle': (0., 0.54, 0.54, 1.), 'square': (0.54, 0., 0.54, 1.), 
	'man-body1': (1., 0., 0., 1.), 'man-body2': (0., 0., 1., 1.), 'man-head1': (0.8, 0., 0.8, 1.), 'man-head2': (0.8, 0., 0.8, 1.)
	}
COLORS_HARMONIC_PHASE = {
	'circle': (0., 0.54, 0.54, 1.), 'square': (0.54, 0., 0.54, 1.),
	'man-body1': (1., 0., 0., 1.), 'man-body2': (0., 0., 1., 1.), 'man-head1': (0.8, 0., 0.8, 1.), 'man-head2': (0.8, 0., 0.8, 1.)
	}
COLORS_DETAIL = {
	'man-detail-common': (0.58, 0., 0.83, 1.), 'man-detail-body1': (0.863, 0.08, 0.24, 1.), 'man-detail-body2': (0.13, 0.7, 0.67, 1.)
}
COLORS_DETAIL_LINES = {
	'man-detail-common': (0.8, 0., 0.8, 1.), 'man-detail-body1': (1., 0., 0., 1.), 'man-detail-body2': (0., 0., 1., 1.)
}
COLOR_AUTOMATA = (0.54, 0., 0., 1.)

TITLE_FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL

ONES_FRAME = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], 4), 1.)

MAX_PERCEPT_RADIUS = 136
FOREHEAD_CENTER = (345, 170)  # xy

IMAGENET_MEAN = torch.tensor([[[0.485, 0.456, 0.406]]]).permute(2,1,0)
IMAGENET_STD = torch.tensor([[[0.229, 0.224, 0.225]]]).permute(2,1,0)
NUM_ITERS = 10
DEVICE = 'cuda'

# Output config
OUTPUT_FRAMES_DIR = "./output/frames"
WRITER_FRAME_SKIP = 4
SCENE_FPS = {'scene_1': 0.2, 'scene_2': 7, 'scene_3': 25, 'scene_4': 5, 'scene_5': 10, 'scene_6': 25, 'scene_7': 10, 'scene_8': 10}


# ---
# Utils

class FramesWriter:

	def __init__(self):
		self.scene = 0
		self.frame_counter = 0
		self.frame_skip = WRITER_FRAME_SKIP
		self.output_dir = OUTPUT_FRAMES_DIR
		os.makedirs(self.output_dir, exist_ok=True)

	def set_scene(self, scene):
		self.scene = scene
		os.makedirs(f"{self.output_dir}/{scene}", exist_ok=True)

	def write(self, frame):
		self.frame_counter += 1
		if self.frame_counter % self.frame_skip == 0:
			filename = str(self.frame_counter).zfill(5)
			plt.imsave(f"{self.output_dir}/{self.scene}/{filename}.png", frame)


class ParticleSwarm:

	def __init__(self, obj, obj_params, num_particles, omega, phi_p, phi_s, init_pos):
		self.obj = obj
		self.obj_params = obj_params
		self.num_particles = num_particles
		self.omega = omega
		self.phi_p = phi_p
		self.phi_s = phi_s
        
		self.p_pos = np.repeat(np.expand_dims(np.array(init_pos), axis=0), num_particles, axis=0) + 50*np.random.randn(num_particles, 2)
		self.p_vel = np.random.rand(num_particles, 2) * 50

		self.p_pos_best = self.p_pos.copy()
		obj_values = np.array([self.obj(p, self.obj_params) for p in self.p_pos])
		self.swarm_best = self.p_pos[ np.argmin(obj_values) ]

	def step(self):

		for p in range(self.num_particles):
			r_p, r_s = np.random.rand(1, 2), np.random.rand(1, 2)
			self.p_vel[p] = self.omega * self.p_vel[p] + \
			                self.phi_p * r_p * (self.p_pos_best[p] - self.p_pos[p]) + \
			                self.phi_s * r_s * (self.swarm_best - self.p_pos[p])
                
			self.p_pos[p] += self.p_vel[p]
			
			if self.obj(self.p_pos[p], self.obj_params) < self.obj(self.p_pos_best[p], self.obj_params):
				self.p_pos_best[p] = self.p_pos[p].copy()
				if self.obj(self.p_pos_best[p], self.obj_params) < self.obj(self.swarm_best, self.obj_params):
					self.swarm_best = self.p_pos_best[p].copy()


class GameOfLife:

	def __init__(self, automata_patterns, mask_internal):		
		self.size = IMAGE_SIZE[1]//4, IMAGE_SIZE[0]//4  # HW
		
		self.state = np.zeros(self.size)

		# Set pattern 'p23 honey farm hassler' having an oscillation period of 23
		pattern_array = automata_patterns['p23honeyfarmhassler']
		offset = (self.size[0]//3 + 0, self.size[1]//2 - 9)		
		self.state[offset[0]:offset[0]+pattern_array.shape[0], offset[1]:offset[1]+pattern_array.shape[1]] = pattern_array
		
		# Set pattern 'twin bees shuttle' having an oscillation period of 46
		pattern_array = automata_patterns['twinbeesshuttle']
		pattern_array = np.flip(pattern_array).T
		offset = (self.size[0]//2 - 3, self.size[1]//2 - 6)
		self.state[offset[0]:offset[0]+pattern_array.shape[0], offset[1]:offset[1]+pattern_array.shape[1]] = pattern_array

		self.alive_kernel = np.array(
			[[1,1,1],
			 [1,0,1],
			 [1,1,1]]
			 )

		self.mask_internal = mask_internal

	def step(self, frame):
		
		# Update state
		num_nbs_alive_map = convolve2d(self.state, self.alive_kernel, mode='same')
		state_new = np.zeros_like(self.state)
		dead_to_alive_mask = np.logical_and(self.state==0, num_nbs_alive_map==3)
		state_new[dead_to_alive_mask] = 1
		alive_to_alive_mask = np.logical_and(self.state==1, np.logical_or(num_nbs_alive_map==2, num_nbs_alive_map==3))
		state_new[alive_to_alive_mask] = 1
		self.state = state_new

		# Commit to frame
		state_viz = resize(self.state, IMAGE_SIZE, order=0)
		dead_cells_mask = np.logical_and(state_viz == 0, self.mask_internal)
		alive_cells_mask = np.logical_and(state_viz == 1, self.mask_internal)
		# dead_cells_mask = np.stack([dead_cells_mask, dead_cells_mask, dead_cells_mask, np.ones_like(dead_cells_mask)], axis=2)
		# alive_cells_mask = np.stack([alive_cells_mask, alive_cells_mask, alive_cells_mask, np.ones_like(alive_cells_mask)], axis=2)
		frame[dead_cells_mask] = COLOR_WHITE
		frame[alive_cells_mask] = COLOR_AUTOMATA

		return frame


class VGG16Backbone(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        net = vgg16(pretrained=True).eval().to(DEVICE)
        self.backbone = net.features
        self.relevant_layer_idxs = [15, 18, 20, 22, 25, 27, 29]

    def forward(self, x):
        features = []
        for layer_idx in range(30):
            x = self.backbone[layer_idx](x)
            if layer_idx in self.relevant_layer_idxs:
                features.append(x)
        return features
    

def load_contours():
	contours = {}
	for key in CONTOUR_PATHS.keys():
		contour = plt.imread(CONTOUR_PATHS[key])
		contour = (1 - contour)
		if key not in ['man-detail-common', 'man-detail-body1', 'man-detail-body2']:
			contour = skeletonize(contour)		
		contours[key] = contour
	return contours


def load_masks():
	masks = {}
	for key in MASK_PATHS.keys():
		mask = plt.imread(MASK_PATHS[key])
		mask = (1 - mask).astype(bool)
		masks[key] = mask
	return masks


def load_art():
	art = {}
	for key in ART_PATHS.keys():
		img = plt.imread(ART_PATHS[key])
		if img.shape[2] == 3: img = np.concatenate([img, np.ones_like(img[:,:,0:1])], axis=2) # Add alpha channel. TODO: remove later.
		art[key] = img
	return art


def load_automata_patterns():
	
	automata_patterns = {}

	for key in AUTOMATA_PATTERN_PATHS.keys():

		with open(AUTOMATA_PATTERN_PATHS[key]) as fs:
			pattern = fs.read()
		pattern_lines = pattern.split('\n')
		pattern_array = np.zeros((len(pattern_lines), len(pattern_lines[0])))		
		for i in range(len(pattern_lines)):
			for j in range(len(pattern_lines[i])):				
				if pattern_lines[i][j] == 'O':   pattern_array[i,j] = 1
				elif pattern_lines[i][j] == '.': pattern_array[i,j] = 0
		automata_patterns[key] = pattern_array
	
	return automata_patterns


def load_portraits():
	portraits = {}
	for key in PORTRAIT_PATHS.keys():
		portrait = plt.imread(PORTRAIT_PATHS[key])
		if portrait.shape[2] == 4:
			portrait = portrait[:,:,:3]
		portrait = resize(portrait, (2*MAX_PERCEPT_RADIUS, 2*MAX_PERCEPT_RADIUS))
		portraits[key] = portrait
	return portraits


def coords_ul2c(x, y, center=(IMAGE_SIZE[1] // 2, IMAGE_SIZE[0] // 2)):
	x = x - center[0]
	y = center[1] - y
	return (round(x), round(y))


def coords_c2ul(x, y):
	x = x + IMAGE_SIZE[0] // 2
	y = IMAGE_SIZE[1] // 2 - y
	return (round(x), round(y))


def algo_walk(contour, origin):
	signal, _ = cv2.findContours(contour.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
	signal = signal[0].squeeze()
	signal = np.array([coords_ul2c(s[0], s[1], origin) for s in signal])
	signal = signal[:,0] + 1j*signal[:,1]
	return signal


def encode_contours_into_signals(contours):
	"""
	Obtain discrete-time signal that encodes the contour.
	"""
	
	signals = {}
	for sn in ['circle', 'square', 'man-body1','man-body2', 'man-head1','man-head2']:
		signals[sn] = algo_walk(contours[sn], SIGNAL_ORIGINS[sn])
	
		if SIGNAL_MODS[sn]['shift'] is not None:

			first_val = coords_ul2c(*SIGNAL_MODS[sn]['shift'], center=SIGNAL_ORIGINS[sn])
			first_val = first_val[0] + 1j*first_val[1]
			first_val_idx = np.squeeze(np.argwhere(signals[sn] == first_val))
			signals[sn] = np.roll(signals[sn], shift=(signals[sn].shape[0] - first_val_idx))
		
		if SIGNAL_MODS[sn]['reverse']:
			signals[sn] = signals[sn][::-1]

	return signals


def dtfs(signal, K):
	"""	Discrete-time Fourier series """
	
	# Discrete-time stuff
	N = signal.shape[0]
	w0 = 2*np.pi / N
	n_vec = np.arange(N)

	spectrum = []
	for k in range(K):
		ak = (1/N) * np.sum(signal * np.exp(-1j * k * w0 * n_vec))
		spectrum.append(ak)

	spectrum = np.array(spectrum)
	return spectrum


def compute_dtfs_of_signals(signals):
	spectra = {}
	for sn in signals.keys():			
		signal = signals[sn]
		# if sn == 'circle':  K = 2
		# else:               K = signal.shape[0]
		spectrum = dtfs(signal, signal.shape[0])
		spectra[sn] = spectrum
	return spectra


def draw_axle(frame, coords):
	cv2.circle(frame, coords, 12, COLOR_BLACK, -1)
	cv2.circle(frame, coords, 10, (0.5, 0.5, 0.5, 1.), -1)
	cv2.circle(frame, coords, 8, COLOR_BLACK, -1)
	cv2.circle(frame, coords, 6, (0.5, 0.5, 0.5, 1.), -1)
	cv2.circle(frame, coords, 4, COLOR_BLACK, -1)
	cv2.circle(frame, coords, 2, (0.5, 0.5, 0.5, 1.), -1)
	return frame


def draw_slider(frame, coords, orientation):
	if orientation == 'horizontal':
		cv2.rectangle(frame, (coords[0]-8, coords[1]-4), (coords[0]+8, coords[1]+4), COLOR_BLACK, -1)
		cv2.rectangle(frame, (coords[0]-6, coords[1]-2), (coords[0]+6, coords[1]+2), (0.5, 0.5, 0.5, 1.), -1)
		cv2.rectangle(frame, (coords[0]-4, coords[1]-1), (coords[0]+4, coords[1]+1), COLOR_BLACK, -1)
		cv2.rectangle(frame, (coords[0]-2, coords[1]-1), (coords[0]+2, coords[1]+1), (0.5, 0.5, 0.5, 1.), -1)
	elif orientation == 'vertical':
		cv2.rectangle(frame, (coords[0]-4, coords[1]-8), (coords[0]+4, coords[1]+8), COLOR_BLACK, -1)
		cv2.rectangle(frame, (coords[0]-2, coords[1]-6), (coords[0]+2, coords[1]+6), (0.5, 0.5, 0.5, 1.), -1)
		cv2.rectangle(frame, (coords[0]-1, coords[1]-4), (coords[0]+1, coords[1]+4), COLOR_BLACK, -1)
		cv2.rectangle(frame, (coords[0]-1, coords[1]-2), (coords[0]+1, coords[1]+2), (0.5, 0.5, 0.5, 1.), -1)
	return frame


def apply_shade(frame, shade, masks, contours):
	mask_bg = np.logical_not(np.logical_or(masks['man'], (contours['circle'] + contours['square']).astype(bool)))
	mask_bg = np.stack([mask_bg, mask_bg, mask_bg, np.ones_like(mask_bg)], axis=2)	
	mask_shade = np.logical_and(shade<1., mask_bg)
	frame[mask_shade] = mask_bg[mask_shade]*shade[mask_shade]
	return frame


def update_image(image, backbone, optimizer):
    optimizer.zero_grad()
    features = backbone(image)

    features = torch.cat([f.flatten() for f in features], dim=0)
    loss = -F.mse_loss(features, torch.zeros_like(features, device=DEVICE))

    loss.backward()
    optimizer.step()
    return image


def postprocess_dream_image(dream_tensor):
    dream_tensor = dream_tensor.detach().cpu()
    dream_tensor = dream_tensor * IMAGENET_STD + IMAGENET_MEAN
    dream_tensor = torch.clip(dream_tensor, 0.0, 1.0)
    image = dream_tensor.squeeze().permute(1,2,0).numpy()
    return image


def inject_portrait(dream_tensor, portrait, alpha):	
	# portrait = preprocess_dream_image(portrait)
	# dream_tensor = alpha*portrait + (1-alpha)*dream_tensor
	
	##

	portrait = torch.tensor(portrait, dtype=torch.float, device=DEVICE).permute(2,0,1).unsqueeze(0)
	dream_tensor = (dream_tensor - dream_tensor.min()) / (dream_tensor.max() - dream_tensor.min())
	dream_tensor = alpha*portrait + (1-alpha)*dream_tensor
	dream_tensor = (dream_tensor - IMAGENET_MEAN.to(DEVICE)) / IMAGENET_STD.to(DEVICE)
	
	return dream_tensor


def draw_text(
		frame, text, pos,
		font=TITLE_FONT,
		font_scale=1,
		font_thickness=1,
		text_color=COLOR_WHITE,
		text_color_bg=(0.2, 0.2, 0.2, 1.)
		):

	x, y = pos
	text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
	text_w, text_h = text_size
	cv2.rectangle(frame, (x-5, y-10), (x + text_w + 5, y + text_h + 10), text_color_bg, -1)
	cv2.putText(frame, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)    

	return frame, text_size



# ---
# Scene implementation

def scene_1(writer):
	"""	Chaos """

	writer.set_scene('scene_1')
	writer.frame_skip = 1

	noise_level = 0.6
	noise = noise_level * np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1])

	frame = ONES_FRAME - np.stack([noise, noise, noise, np.zeros_like(noise)], axis=2)
	frame_with_text, _ = draw_text(frame.copy(), text="1. Chaos", pos=(20, 20), font=TITLE_FONT)
	writer.write(frame_with_text)

	return noise


def scene_2(noise, writer):
	""" Fluctuation and First Automata """

	writer.set_scene('scene_2')
	writer.frame_skip = 1


	def obj_display(xx, yy, params):

		pts = np.stack((yy.flatten(), xx.flatten()), axis=1)

		cov1 = np.array([[params['d1']*100, -90], [-90, params['d1']*900]])
		mean1 = np.array([100, 200])
		pocket1 = multivariate_normal.pdf(pts, mean=mean1, cov=cov1).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])

		cov2 = np.array([[params['d2']*900, 50], [50, params['d2']*100]])
		mean2 = np.array([200, 500])
		pocket2 = multivariate_normal.pdf(pts, mean=mean2, cov=cov2).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])

		cov3 = np.array([[params['d3']*100, 50], [50, params['d3']*900]])
		mean3 = np.array([500, 400])
		pocket3 = multivariate_normal.pdf(pts, mean=mean3, cov=cov3).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
		
		covc = np.array([[1000, 0], [0, 1000]])
		meanc= np.array([IMAGE_SIZE[1]//2, IMAGE_SIZE[0]//2])
		pocketc = multivariate_normal.pdf(pts, mean=meanc, cov=covc).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])

		cost = pocket1  + pocket2 + pocket3 + params['wc']*pocketc
		cost = -cost
		cost = (cost - cost.min()) / (cost.max() - cost.min())
		return cost

	def obj_swarm(pt, params):

		cov1 = np.array([[params['d1']*100, -90], [-90, params['d1']*900]])
		mean1 = np.array([100, 200])
		pocket1 = multivariate_normal.pdf(pt, mean=mean1, cov=cov1)

		cov2 = np.array([[params['d2']*900, 50], [50, params['d2']*100]])
		mean2 = np.array([200, 500])
		pocket2 = multivariate_normal.pdf(pt, mean=mean2, cov=cov2)

		cov3 = np.array([[params['d3']*100, 50], [50, params['d3']*900]])
		mean3 = np.array([500, 400])
		pocket3 = multivariate_normal.pdf(pt, mean=mean3, cov=cov3)
		
		# center_pocket = np.sqrt((pt[1] - IMAGE_SIZE[1]//2)**2 + (pt[0] - IMAGE_SIZE[0]//2)**2)
		covc = np.array([[10000, 0], [0, 10000]])
		meanc= np.array([IMAGE_SIZE[1]//2, IMAGE_SIZE[0]//2])
		pocketc = multivariate_normal.pdf(pt, mean=meanc, cov=covc)

		cost = pocket1 + pocket2 + pocket3 + params['wc']*pocketc
		return -cost


	# Pockets appear
	xx, yy = np.meshgrid(np.arange(IMAGE_SIZE[1]), np.arange(IMAGE_SIZE[0]))
	for alpha in np.linspace(0, 1, 3*SCENE_FPS['scene_2']):
		landscape = obj_display(xx, yy, params={'d1':1,'d2': 1, 'd3': 1, 'wc': 0})	
		frame = (1-alpha)*noise + alpha*noise*landscape
		frame = ONES_FRAME - np.stack([frame, frame, frame, np.zeros_like(frame)], axis=2)
		frame_with_text, _ = draw_text(frame.copy(), text="2. Fluctuations and First Automata", pos=(20, IMAGE_SIZE[1]-40))
		writer.write(frame_with_text)


	# Life emerges	
	swarm1 = ParticleSwarm(obj_swarm, {'d1':1, 'd2':1, 'd3':1, 'wc': 0}, 100, omega=0.9, phi_p=0.2, phi_s=0.1, init_pos=[100, 200])
	swarm2 = ParticleSwarm(obj_swarm, {'d1':1, 'd2':1, 'd3':1, 'wc': 0}, 100, omega=0.9, phi_p=0.2, phi_s=0.1, init_pos=[500, 400])

	num_obj_steps = 100
	d1 = np.linspace(1, 50, num_obj_steps)
	d2 = np.linspace(1, 50, num_obj_steps)
	d3 = np.linspace(1, 50, num_obj_steps)
	wc = np.linspace(0, 10, num_obj_steps)
	
	for i in tqdm(range(num_obj_steps)):
		
		landscape = obj_display(xx, yy, params={'d1': d1[i],'d2': d2[i], 'd3': d3[i], 'wc': wc[i]})
		frame = noise * landscape
		frame = ONES_FRAME - np.stack([frame, frame, frame, np.zeros_like(frame)], axis=2)
		frame_copy = frame.copy()

		swarm1.obj_params = {'d1': d1[i],'d2': d2[i], 'd3': d3[i], 'wc': wc[i]}
		swarm2.obj_params = {'d1': d1[i],'d2': d2[i], 'd3': d3[i], 'wc': wc[i]}		
		swarm1.step()
		swarm2.step()
		for p in swarm1.p_pos:
			color = np.array((0., 0.54, 0.54, 0.8))
			if np.random.random() < 0.5:  # Mutate
				color[:3] = color[:3] + (np.random.random(3) - 0.5)
				color[:3] = np.clip(color[:3], 0., 1.)
			cv2.circle(frame_copy, (round(p[1]), round(p[0])), 4, tuple(color), -1)
		for p in swarm2.p_pos:
			color = np.array((0.54, 0., 0.54, 1.))
			if np.random.random() < 0.5:  # Mutate
				color[:3] = color[:3] + (np.random.random(3) - 0.5)
				color[:3] = np.clip(color[:3], 0., 1.)
			cv2.circle(frame_copy, (round(p[1]), round(p[0])), 4, tuple(color), -1)
		
		frame_with_text, _ = draw_text(frame_copy.copy(), text="2. Fluctuations and First Automata", pos=(20, IMAGE_SIZE[1]-40))
		writer.write(frame_with_text)
		if i == 0:  
			for _ in range(3*SCENE_FPS['scene_2']): writer.write(frame_with_text)

	# Let both swarms fully converge
	swarm1.omega -= 0.1
	for step in range(20):
		frame_copy = frame.copy()
		swarm1.step()
		swarm2.step()
		for p in swarm1.p_pos:
			color = np.array((0., 0.54, 0.54, 0.8))
			if np.random.random() < 0.5:  # Mutate
				color[:3] = color[:3] + (np.random.random(3) - 0.5)
				color[:3] = np.clip(color[:3], 0., 1.)
			cv2.circle(frame_copy, (round(p[1]), round(p[0])), 4, tuple(color), -1)
		for p in swarm2.p_pos:
			color = np.array((0.54, 0., 0.54, 1.))
			if np.random.random() < 0.5:  # Mutate
				color[:3] = color[:3] + (np.random.random(3) - 0.5)
				color[:3] = np.clip(color[:3], 0., 1.)
			cv2.circle(frame_copy, (round(p[1]), round(p[0])), 4, tuple(color), -1)

		frame_with_text, _ = draw_text(frame_copy.copy(), text="2. Fluctuations and First Automata", pos=(20, IMAGE_SIZE[1]-40))
		writer.write(frame_with_text)

	for _ in range(2*SCENE_FPS['scene_2']): writer.write(frame_with_text)
	return frame


def scene_3(frame, spectra, signals, writer):
	""" Paradigm shift """

	writer.set_scene('scene_3')
	writer.frame_skip = 1

	frame_with_axle = frame.copy()
	draw_axle(frame_with_axle, SIGNAL_ORIGINS['circle'])
	frame_with_axle_and_text, _ = draw_text(frame_with_axle.copy(), text="3. Paradigm Shift", pos=(20, 20))
	for _ in range(3*SCENE_FPS['scene_3']): writer.write(frame_with_axle_and_text)

	N_max = signals['square'].shape[0]
	n_vec = np.arange(N_max) # indices
	writer.frame_skip = 10

	# Run
	for n in tqdm(n_vec):  # For each time step

		frame_copy = frame_with_axle.copy()

		for sn in ['square', 'circle']:

			N = signals[sn].shape[0]
			if N < (N_max - n):
				continue
			ni = N - (N_max - n)

			w0 = 2*np.pi / N  # Fundamental frequency

			origin = SIGNAL_ORIGINS[sn]  
			spectrum = spectra[sn] 			
			color_line = COLORS_HARMONIC_PHASE[sn]
			color_circle = COLORS_HARMONIC_MAG[sn]
			color_estim = COLORS_STRUCT_ESTIM[sn]
			
			cx, cy = coords_ul2c(*origin)

			for k in range(0, len(spectrum)):  # For each harmonic
				
				magnitude = np.abs(spectrum[k])
				phase = np.angle(spectrum[k])
				pos = magnitude * np.exp(1j * (k*w0*ni + phase))

				kx, ky = pos.real, pos.imag
				kx, ky = cx + kx, cy + ky
				cv2.line(frame_copy, coords_c2ul(cx, cy), coords_c2ul(kx, ky), color_line, 1)
				if sn != 'circle':  # Don't draw any circles if the contour itself is circle
					cv2.circle(frame_copy, coords_c2ul(cx, cy), round(magnitude), color_circle, 1)
				cx, cy = kx, ky

				# Draw signal estimate
				if k == len(spectrum) - 1:
					cv2.circle(frame_copy, coords_c2ul(kx, ky), 3, (0., 0., 0., 1.), -1)
					x, y = coords_c2ul(kx, ky)
					frame_with_axle[y, x] = color_estim
					frame[y, x] = color_estim

		frame_copy_with_text, _ = draw_text(frame_copy.copy(), text="3. Paradigm Shift", pos=(20, 20))
		writer.write(frame_copy_with_text)

	writer.frame_skip = 1
	frame_with_axle_and_text, _ = draw_text(frame_with_axle.copy(), text="3. Paradigm Shift", pos=(20, 20))
	frame_with_text, _ = draw_text(frame.copy(), text="3. Paradigm Shift", pos=(20, 20))
	for _ in range(3*SCENE_FPS['scene_3']): writer.write(frame_with_axle_and_text)
	for _ in range(3*SCENE_FPS['scene_3']): writer.write(frame_with_text)
	return frame


def scene_4(frame, masks, contours, writer):
	""" Habitat """

	writer.set_scene('scene_4')
	writer.frame_skip = 1

	mask_square = dilation(masks['square'], square(5))
	mask_square = np.logical_xor(mask_square, np.logical_and(mask_square, contours['square']))
	mask_square = np.logical_xor(mask_square, np.logical_and(mask_square, contours['circle']))
	
	mask_circle = dilation(masks['circle'], disk(4))
	mask_circle = np.logical_xor(mask_circle, np.logical_and(mask_circle, contours['circle']))
	mask_circle = np.logical_xor(mask_circle, np.logical_and(mask_circle, contours['square']))
	
	mask = np.logical_or(mask_square, mask_circle)
	mask = np.stack([mask, mask, mask, np.ones_like(mask)], axis=2)

	frame_copy = frame.copy()
	for alpha in np.linspace(0, 0.8, 3*SCENE_FPS['scene_4']):
		frame_copy[mask] = (1-alpha)*frame[mask] + alpha
		frame_copy_with_text, _ = draw_text(frame_copy.copy(), text="4. Habitat", pos=(20, 20))
		writer.write(frame_copy_with_text)
	frame = frame_copy
	return frame


def scene_5(frame, spectra, signals, writer):
	""" Harmonic synthesis """

	writer.set_scene('scene_5')	

	frame_with_axle = frame.copy()
	draw_axle(frame_with_axle, SIGNAL_ORIGINS['man-body1'])
	draw_axle(frame_with_axle, SIGNAL_ORIGINS['man-body2'])
	draw_axle(frame_with_axle, SIGNAL_ORIGINS['man-head1'])
	
	writer.frame_skip = 1
	frame_with_text, _ = draw_text(frame.copy(), text="5. Harmonic", pos=(20, 20))
	frame_with_text, _ = draw_text(frame_with_text, text="Synthesis", pos=(55, 50))
	frame_with_axle_and_text, _ = draw_text(frame_with_axle.copy(), text="5. Harmonic", pos=(20, 20))
	frame_with_axle_and_text, _ = draw_text(frame_with_axle_and_text, text="Synthesis", pos=(55, 50))
	for _ in range(1*SCENE_FPS['scene_5']): writer.write(frame_with_text)
	for _ in range(3*SCENE_FPS['scene_5']): writer.write(frame_with_axle_and_text)

	# Run
	N_max = signals['man-body1'].shape[0]
	n_vec = np.arange(N_max) # indices
	writer.frame_skip = 4
	for n in tqdm(n_vec):  # For each time step

		frame_copy = frame_with_axle.copy()

		for sn in ['man-body1', 'man-body2', 'man-head1', 'man-head2']:

			N = signals[sn].shape[0]
			if N < (N_max - n):
				continue
			ni = N - (N_max - n)

			w0 = 2*np.pi / N  # Fundamental frequency

			origin = SIGNAL_ORIGINS[sn]  
			spectrum = spectra[sn] 			
			color_line = COLORS_HARMONIC_PHASE[sn]
			color_circle = COLORS_HARMONIC_MAG[sn]
			color_estim = COLORS_STRUCT_ESTIM[sn]
			
			cx, cy = coords_ul2c(*origin)

			for k in range(0, len(spectrum)):  # For each harmonic
				
				magnitude = np.abs(spectrum[k])
				phase = np.angle(spectrum[k])
				pos = magnitude * np.exp(1j * (k*w0*ni + phase))

				kx, ky = pos.real, pos.imag
				kx, ky = cx + kx, cy + ky
				cv2.line(frame_copy, coords_c2ul(cx, cy), coords_c2ul(kx, ky), color_line, 1)
				cv2.circle(frame_copy, coords_c2ul(cx, cy), round(magnitude), color_circle, 1)
				cx, cy = kx, ky

				# Draw signal estimate
				if k == len(spectrum) - 1:
					cv2.circle(frame_copy, coords_c2ul(kx, ky), 4, COLOR_BLACK, -1)
					x, y = coords_c2ul(kx, ky)
					if frame[y, x, 0] == frame[y, x, 1] and frame[y, x, 1] == frame[y, x, 2]:  # if no color has been added here, add native body color
						frame_with_axle[y, x] = color_estim
						frame[y, x] = color_estim
					else:   # Else, add the common color
						frame_with_axle[y, x] = COLORS_STRUCT_ESTIM['man-head1']
						frame[y, x] = COLORS_STRUCT_ESTIM['man-head1']

		frame_copy_with_text, _ = draw_text(frame_copy.copy(), text="5. Harmonic", pos=(20, 20))
		frame_copy_with_text, _ = draw_text(frame_copy_with_text, text="Synthesis", pos=(55, 50))
		writer.write(frame_copy_with_text)

	writer.frame_skip = 1
	frame_with_text, _ = draw_text(frame.copy(), text="5. Harmonic", pos=(20, 20))
	frame_with_text, _ = draw_text(frame_with_text, text="Synthesis", pos=(55, 50))
	frame_with_axle_and_text, _ = draw_text(frame_with_axle.copy(), text="5. Harmonic", pos=(20, 20))
	frame_with_axle_and_text, _ = draw_text(frame_with_axle_and_text, text="Synthesis", pos=(55, 50))
	for _ in range(2*SCENE_FPS['scene_5']): writer.write(frame_with_axle_and_text)
	for _ in range(3*SCENE_FPS['scene_5']): writer.write(frame_with_text)
	return frame


def scene_6(frame, contours, writer):
	"""	Cartesian refinement """

	writer.set_scene('scene_6')
	writer.frame_skip = 4

	for sn in ['man-detail-body1', 'man-detail-body2', 'man-detail-common']:
		
		labelmap = measure.label(contours[sn], background=0)
		label_ids = np.unique(labelmap)
		
		# Get connected components (aka "details")
		details = []
		for label_id in label_ids:
			if label_id == 0: continue
			detail = np.argwhere(labelmap == label_id)
			detail = list(zip(detail[:, 0], detail[:, 1]))
			details.append(detail)

		# Parallely draw at most 3 details
		random.shuffle(details)
		buffer = []
		idxs = np.random.choice(len(details), 3, replace=False)
		for idx in idxs: buffer.append(details[idx])
		for idx in idxs: details.pop(idx)

		while True:

			if len(buffer) < 3 and len(details) > 0:
				idxs = np.random.choice(len(details), min(3-len(buffer), len(details)), replace=False)
				for idx in idxs: buffer.append(details[idx])
				for idx in sorted(idxs, reverse=True): details.pop(idx)
				
			elif len(buffer) == 0 and len(details) == 0:
				break

			to_pop = []
			frame_copy = frame.copy()
			for d, detail in enumerate(buffer):

				if len(detail) == 0: 
					to_pop.append(d)
					continue

				y,x = detail.pop(0)
				cv2.line(frame_copy, (x, 143), (x, 656), COLORS_DETAIL_LINES[sn], 1) # vert line
				cv2.line(frame_copy, (86, y), (605, y), COLORS_DETAIL_LINES[sn], 1)  # horiz line
			
				frame_copy = draw_slider(frame_copy, (x, 143), orientation='horizontal') # slider
				frame_copy = draw_slider(frame_copy, (x, 656), orientation='horizontal') #
				frame_copy = draw_slider(frame_copy, (86, y), orientation='vertical')    #
				frame_copy = draw_slider(frame_copy, (605, y), orientation='vertical')   #

				cv2.circle(frame_copy, (x, y), 4, COLOR_BLACK, -1)  # Marker
				frame[y, x] = COLORS_DETAIL[sn]

			for d in sorted(to_pop, reverse=True): buffer.pop(d)

			frame_copy_with_text, _ = draw_text(frame_copy.copy(), text="6. Cartesian Refinement", pos=(20, 20))
			writer.write(frame_copy_with_text)

	writer.frame_skip = 1
	frame_with_text, _ = draw_text(frame.copy(), text="6. Cartesian Refinement", pos=(20, 20))
	for _ in range(2*SCENE_FPS['scene_6']): writer.write(frame_with_text)
	return frame


def scene_7(frame, contours, masks, art, automata_patterns, writer):
	"""	Homeostasis """

	writer.set_scene('scene_7')
	writer.frame_skip = 1


	# Shade to emphasize the body		
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(SCENE_FPS['scene_7']):  writer.write(frame_with_text)

	frame = apply_shade(frame, art['shade1'], masks, contours)
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(int(0.5*SCENE_FPS['scene_7'])):  writer.write(frame_with_text)
	
	frame = apply_shade(frame, art['shade2'], masks, contours)
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(int(0.5*SCENE_FPS['scene_7'])):  writer.write(frame_with_text)
	
	frame = apply_shade(frame, art['shade3'], masks, contours)
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(int(0.5*SCENE_FPS['scene_7'])):  writer.write(frame_with_text)
	
	frame = apply_shade(frame, art['shade4'], masks, contours)
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(int(0.5*SCENE_FPS['scene_7'])):  writer.write(frame_with_text)
	
	frame = apply_shade(frame, art['shade5'], masks, contours)
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(int(0.5*SCENE_FPS['scene_7'])):  writer.write(frame_with_text)
	
	frame = apply_shade(frame, art['shade6'], masks, contours)
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(int(0.5*SCENE_FPS['scene_7'])):  writer.write(frame_with_text)
		

	# Remove entropy from the body
	body_contour = contours['man-body1'] + contours['man-body2'] + contours['man-head1'] + contours['man-head2'] + \
	               contours['man-detail-common'] + contours['man-detail-body1'] + contours['man-detail-body2']
	mask_internal = np.logical_xor(masks['man'], body_contour.astype(bool))	
	frame[mask_internal] = 1.0
	frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
	for _ in range(1*SCENE_FPS['scene_7']):  writer.write(frame_with_text)


	# Game of life
	automata = GameOfLife(automata_patterns, mask_internal)
	for step in tqdm(range(8*SCENE_FPS['scene_7'])):		
		frame = automata.step(frame)
		frame_with_text, _ = draw_text(frame.copy(), text="7. Homeostasis", pos=(20, 20))
		if step == 0:
			for _ in range(3*SCENE_FPS['scene_7']):  writer.write(frame_with_text)
		else:
			writer.write(frame_with_text)

	return frame, automata


def scene_8(frame, automata, masks, portraits, writer):
	"""	Abstraction """

	writer.set_scene('scene_8')
	writer.frame_skip = 1	

	# The growing sphere of perception
	offset = (FOREHEAD_CENTER[1] - MAX_PERCEPT_RADIUS, FOREHEAD_CENTER[0] - MAX_PERCEPT_RADIUS)
	mask_head = masks['man-head'][offset[0] : offset[0] + 2*MAX_PERCEPT_RADIUS, offset[1] : offset[1] + 2*MAX_PERCEPT_RADIUS]
	automata.mask_internal = np.logical_xor(automata.mask_internal, np.logical_and(automata.mask_internal, masks['man-head']))

	for r in tqdm(np.arange(1, MAX_PERCEPT_RADIUS-2, 1)):				

		percept = np.random.randn(2*MAX_PERCEPT_RADIUS, 2*MAX_PERCEPT_RADIUS, 4)		
		percept[:,:,3] = 1.0
		
		mask_percept = cv2.circle(np.ones((2*MAX_PERCEPT_RADIUS, 2*MAX_PERCEPT_RADIUS)), (FOREHEAD_CENTER[0]-offset[1], FOREHEAD_CENTER[1]-offset[0]), r, 0.0, -1) == 0.0
		mask_percept = np.logical_xor(mask_percept, np.logical_and(mask_percept, mask_head))	
		
		frame_copy = frame.copy()
		frame_patch = frame_copy[offset[0] : offset[0] + 2*MAX_PERCEPT_RADIUS, offset[1] : offset[1] + 2*MAX_PERCEPT_RADIUS]
		frame_patch[mask_percept] = percept[mask_percept]
		frame_patch = cv2.circle(frame_patch, (FOREHEAD_CENTER[0]-offset[1], FOREHEAD_CENTER[1]-offset[0]), r+1, 0.0, 2)
		
		frame_copy[offset[0] : offset[0] + 2*MAX_PERCEPT_RADIUS, offset[1] : offset[1] + 2*MAX_PERCEPT_RADIUS] = frame_patch
		
		# Automata update and write
		mask_percept_fullsize = np.zeros(IMAGE_SIZE)
		mask_percept_fullsize[offset[0] : offset[0] + 2*MAX_PERCEPT_RADIUS, offset[1] : offset[1] + 2*MAX_PERCEPT_RADIUS] = mask_percept
		mask_percept_fullsize = dilation(mask_percept_fullsize, disk(6))			
		automata.mask_internal = np.logical_xor(automata.mask_internal, np.logical_and(automata.mask_internal, mask_percept_fullsize))
		frame_copy = automata.step(frame_copy)
		frame_copy = np.clip(frame_copy, 0., 1.)
		frame_copy_with_text, _ = draw_text(frame_copy.copy(), text="8. Abstraction", pos=(20, 20))
		writer.write(frame_copy_with_text)	

	frame = frame_copy
	del frame_copy


	# Deep dream
	backbone = VGG16Backbone()
	dream_tensor = torch.tensor(percept[:,:,:3], dtype=torch.float, device=DEVICE)
	dream_tensor = torch.permute(dream_tensor, (2,0,1)).unsqueeze(0)

	zoom_factor = 1.04
	num_zooms = 275
	for n in tqdm(range(num_zooms)):

		dream_tensor = TF.resize(dream_tensor, int(2*MAX_PERCEPT_RADIUS*zoom_factor)).detach()
		dream_tensor = TF.center_crop(dream_tensor, (2*MAX_PERCEPT_RADIUS, 2*MAX_PERCEPT_RADIUS))

		dream_tensor.requires_grad = True
		optimizer = torch.optim.Adam(params=[dream_tensor], lr=1e-2)

		for _ in range(15):
			dream_tensor = update_image(dream_tensor, backbone, optimizer)
		
		if n == 75:  dream_tensor = inject_portrait(dream_tensor, portraits['buddha'], alpha=0.5)
		if n == 100: dream_tensor = inject_portrait(dream_tensor, portraits['plato'], alpha=0.6)
		if n == 125: dream_tensor = inject_portrait(dream_tensor, portraits['davinci'], alpha=0.5)
		if n == 150: dream_tensor = inject_portrait(dream_tensor, portraits['descartes'], alpha=0.5)
		if n == 175: dream_tensor = inject_portrait(dream_tensor, portraits['newton'], alpha=0.6)
		if n == 200: dream_tensor = inject_portrait(dream_tensor, portraits['darwin'], alpha=0.6)
		if n == 225: dream_tensor = inject_portrait(dream_tensor, portraits['jung'], alpha=0.6)
		if n == 250: dream_tensor = inject_portrait(dream_tensor, portraits['turing'], alpha=0.5)

		percept = postprocess_dream_image(dream_tensor)
		percept = np.concatenate([percept, np.ones((2*MAX_PERCEPT_RADIUS, 2*MAX_PERCEPT_RADIUS, 1))], axis=2)

		frame_patch = frame[offset[0] : offset[0] + 2*MAX_PERCEPT_RADIUS, offset[1] : offset[1] + 2*MAX_PERCEPT_RADIUS]
		frame_patch[mask_percept] = percept[mask_percept]

		frame[offset[0] : offset[0] + 2*MAX_PERCEPT_RADIUS, offset[1] : offset[1] + 2*MAX_PERCEPT_RADIUS] = frame_patch

		# Automata update and write
		frame = automata.step(frame)
		frame = np.clip(frame, 0., 1.)
		frame_with_text, _ = draw_text(frame.copy(), text="8. Abstraction", pos=(20, 20))
		writer.write(frame_with_text)

	for _ in range(4*SCENE_FPS['scene_8']):  writer.write(frame_with_text)
	return