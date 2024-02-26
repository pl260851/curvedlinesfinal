import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Create a mask with the same dimensions as the image, initialized to zero (black).
    mask = np.zeros_like(img)
    # Fill the polygon defined by `vertices` with white color (255) on the mask.
    cv2.fillPoly(mask, [vertices], 255)
    # Apply the mask to the image using bitwise AND, keeping only the region of interest.
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_image_for_curves(img, roi_vertices):
    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth the image and reduce noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection to find edges in the blurred image.
    canny_image = cv2.Canny(blurred, 100, 35)
    # Apply the region of interest mask to the Canny image to focus on relevant edges.
    masked_canny = region_of_interest(canny_image, roi_vertices)
    return masked_canny
