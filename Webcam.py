import cv2
from PIL import Image as I
from PIL import ImageEnhance
import numpy as np

class Webcam:

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.once = 0

    def get_frame(self):

        if self.video.isOpened():    
            rval, frame_raw = self.video.read()

        # cv2.normalize(frame_raw, frame_raw, 300, -30, cv2.NORM_MINMAX)
        # cv2.normalize(frame_raw, frame_raw, 350, -45, cv2.NORM_MINMAX)

        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        pil_img = I.fromarray(frame.astype('uint8'), "RGB")

        brightness_obj = ImageEnhance.Brightness(pil_img)
        contrast_obj = ImageEnhance.Contrast(pil_img)
        color_obj = ImageEnhance.Color(pil_img)
        sharpness_obj = ImageEnhance.Color(pil_img)

        pil_img = brightness_obj.enhance(1.6)
        # pil_img = contrast_obj.enhance(0.5)
        # pil_img = color_obj.enhance(1.5)
        # pil_img = sharpness_obj.enhance(1.5)


        # if(self.once==0):
        #     self.once = 1
        #     pil_img.show()

        return frame_raw, pil_img
