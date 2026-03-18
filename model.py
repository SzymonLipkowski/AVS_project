import cv2
import numpy as np


def predict(image):
    IG_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    IG_norm=cv2.normalize(IG_gray,None,0,255,cv2.NORM_MINMAX)
    mask = (IG_norm > 155).astype(np.uint8) * 255
    mask=cv2.medianBlur(mask,15)
    kernel=np.ones((9,9),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return mask