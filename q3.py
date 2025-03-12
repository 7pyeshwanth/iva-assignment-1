# q3.py - Image Smoothing
import cv2

def image_smoothing(image_path):
    image = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite("images/q3_smoothed.png", blurred)

image_smoothing("images/input_image.png")
