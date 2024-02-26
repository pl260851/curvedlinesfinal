import cv2
import numpy as np

#https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html

def region_of_interest(img, vertices):
    # Create a mask
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    # Apply the mask to the image, keeping only the region of interest.
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_image_for_curves(img, roi_vertices):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth the image and reduce noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection to find edges 
    canny_image = cv2.Canny(blurred, 100, 35)
    # Apply the region of interest mask to the Canny image
    masked_canny = region_of_interest(canny_image, roi_vertices)
    return masked_canny
