
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2lab, lab2rgb

from FrameDisplay import FrameDisplay
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
def chosen_frame(video_path, starting_index=0):
    frame_index = 0
    video = cv2.VideoCapture(video_path)
    while frame_index< starting_index:
        ret, frame = video.read()
        # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            break  # No more frames in the video
        frame_index += 1
    return frame



video_path = 'Thermo_5.mp4'
frame = chosen_frame(video_path, 1000)

original = frame.copy()

#
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(img, 5)
_, img_th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
# img_th = np.abs(img_th - 255)
# cnts = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
# color_img = np.stack((img,) * 3, axis=-1)
# min_area = 15
# bats = []
# for c in cnts:
#     area = cv2.contourArea(c)
#     print(area)
#     if area > min_area:
#         cv2.drawContours(color_img, [c], -1, (0, 255, 0), 5)
#         bats.append(c)

cv2.imshow('original', original)
cv2.imshow('contours', img_th)
cv2.waitKey(0)
cv2.destroyAllWindows()