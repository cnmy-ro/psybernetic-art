__author__ = "Chinmay Rao"


import cv2
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips



SCENE_FPS = {'scene_1': 0.2, 'scene_2': 7, 'scene_3': 25, 'scene_4': 5, 'scene_5': 10, 'scene_6': 25, 'scene_7': 10, 'scene_8': 10}
IMAGE_SIZE = (691, 691)

for scene in SCENE_FPS.keys():
    
    print(scene)
    
    video_writer = cv2.VideoWriter(f'./output/videos/{scene}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), SCENE_FPS[scene], IMAGE_SIZE)

    paths = sorted(glob.glob(f'./output/frames/{scene}/*.png'))
    for i in range(len(paths)):
        frame = cv2.imread(paths[i])
        video_writer.write(frame)
    video_writer.release()


videos = [VideoFileClip(f'./output/videos/{scene}.mp4') for scene in SCENE_FPS.keys()]
combined_video = concatenate_videoclips(videos, method="compose")
combined_video.write_videofile(f'./output/videos/eoc.mp4')



# Add audio track via:
# ffmpeg -i eoc.mp4 -i monolith.mp3 -c:v copy -c:a aac -shortest eoc_w_sound.mp4