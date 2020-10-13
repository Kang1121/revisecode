import cv2
import pysift

image = cv2.imread('E:/touxiang.JPG', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)

print(keypoints, descriptors)

