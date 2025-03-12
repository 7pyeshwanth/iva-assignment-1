# q2.py - T-Pyramid Computation
import cv2

def compute_t_pyramid(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    for i in range(4):
        image = cv2.pyrDown(image)
        cv2.imwrite(f"images/q2_level{i+1}.png", image)

compute_t_pyramid("images/input_image.png")
