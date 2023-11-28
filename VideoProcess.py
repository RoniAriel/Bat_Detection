import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageProcessor:
    def __init__(self, video_path, starting_index=0):
        self.video_path = video_path
        self.starting_index = starting_index
        self.frame = self.get_chosen_frame()
        self.original_frame = self.frame.copy()
        self.binary_frame = None
        self.inverted_binary = None
        self.origin_but_cropped = None
        self.sure_bg = None
        self.sure_fg=None

    def get_chosen_frame(self):
        frame_index = 0
        video = cv2.VideoCapture(self.video_path)
        while frame_index < self.starting_index:
            ret, frame = video.read()
            if not ret:
                break  # No more frames in the video
            frame_index += 1
        return frame

    def preprocess_frame(self, threshold_value=30):
        self.frame = self.frame[0:450, 0:640]
        self.origin_but_cropped = self.frame.copy()

        # Convert to grayscale
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.frame = cv2.medianBlur(self.frame, 5)
        self.remove_white()

    def remove_white(self,threshold_value=40):
        # # Create a mask where pixels above the threshold are set to 255 (white)
        mask = (self.frame > threshold_value).astype(np.uint8) * 255
        # Apply the mask to the original grayscale image
        self.frame[mask == 255] = 255
        self.Adaptive_histogram_equalization()

    def Adaptive_histogram_equalization(self):
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20,20))
        # # Apply CLAHE
        self.frame = clahe.apply(self.frame)
        self.Smoothing()

    def Smoothing(self):
        self.frame = cv2.medianBlur(self.frame, 3)
        self.Binarization()

    def Binarization(self, threshold_value=40):
        _, self.binary_frame = cv2.threshold(self.frame,threshold_value, 255, cv2.THRESH_BINARY)
        self.inverted_binary = cv2.bitwise_not(self.binary_frame)
        self.Morphological_Operations()

    def Morphological_Operations(self):
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        opened_frame = cv2.morphologyEx(self.inverted_binary, cv2.MORPH_OPEN, kernel2, iterations=1)
        big_kernel = np.ones((4, 4), np.uint8)        # Adjust the kernel size as needed
        kernel_size = (1, 3)                          # Adjust the size based on your needs
        elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        self.sure_bg =opened_frame
        # self.sure_bg = cv2.dilate(opened_frame, elliptical_kernel, iterations=1)
        distance_transform = cv2.distanceTransform(self.sure_bg, cv2.DIST_L2, 5)
        ret, self.sure_fg = cv2.threshold(distance_transform, 0.6 * distance_transform.max(), 255, 0)
        self.sure_fg = np.uint8(self.sure_fg)
        self.watershed()

    def watershed(self):

        unknown = cv2.subtract(self.sure_bg, self.sure_fg)
        ret, markers = cv2.connectedComponents(self.sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(self.origin_but_cropped, markers)
        self.labels(markers)

    def labels(self, markers):
        labels = np.unique(markers)
        bats = []
        for label in labels[2:]:
                # Create a binary image in which only the area of the label is in the foreground
                # and the rest of the image is in the background
            target = np.where(markers == label, 255, 0).astype(np.uint8)
            kernel = np.ones((10, 10), np.uint8)
            target = cv2.dilate(target, kernel, iterations=1)
                # Perform contour extraction on the created binary image
            contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bats.append(contours[0])
        spacing = 5
                # Draw the outline
        self.origin_but_cropped = cv2.drawContours(self.origin_but_cropped, bats, -1, color=(0, 223, 0), thickness=1)
        self.overlay()

    def overlay(self):
        self.original_frame[0:self.origin_but_cropped.shape[0],0: self.origin_but_cropped.shape[1]] = self.origin_but_cropped
        self.plot_results()


    def plot_results(self):
        cv2.imshow('result', self.original_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_results(self ,file_name):
        cv2.imwrite(file_name, self.original_frame)


# Example usage
video_path = 'Thermo_5.mp4'
processor = ImageProcessor(video_path, starting_index=1000)
processor.preprocess_frame()
file_name = "frame1.jpg"
processor.save_results(file_name)
