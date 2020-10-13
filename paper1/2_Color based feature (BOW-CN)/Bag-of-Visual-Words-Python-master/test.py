import scipy.io as sio
from im2c import im2c
import cv2


data = sio.loadmat('C:/Users/yk/Desktop/VisualTracking/ColorNaming/w2c.mat')
im = cv2.imread('C:/Users/yk/Desktop/VisualTracking/ColorNaming/00000009.jpg')
# print(im.shape)
out = im2c(im, data, 0)

