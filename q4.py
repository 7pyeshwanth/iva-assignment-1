# q4.py - Edge Detection (Sobel & Canny)
import cv2
import numpy as np

def edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    canny = cv2.Canny(image, 100, 200)
    cv2.imwrite("images/q4_sobelx.png", sobelx)
    cv2.imwrite("images/q4_sobely.png", sobely)
    cv2.imwrite("images/q4_canny.png", canny)

edge_detection("images/input_image.png")
