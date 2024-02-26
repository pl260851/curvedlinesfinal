import cv2

def draw_midline(img, contours):
    # Check if exactly two contours are provided. If not, exit the function.
    if len(contours) != 2:
        return

    # Calculate midpoints between the corresponding points of the two contours.
    midpoints = []
    for point_set_1, point_set_2 in zip(contours[0], contours[1]):
        midpoint = ((point_set_1[0][0] + point_set_2[0][0]) // 2, (point_set_1[0][1] + point_set_2[0][1]) // 2)
        midpoints.append(midpoint)

    # Draw lines connecting the midpoints to form the midline.
    for i in range(len(midpoints) - 1):
        cv2.line(img, midpoints[i], midpoints[i + 1], (255, 255, 0), 2)

def find_and_draw_contours(img, canny_image):
    # Find contours in the Canny image.
    contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area and keep the two largest ones.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    # Draw a midline between the two largest contours.
    draw_midline(img, contours)
    # Draw the contours on the image.
    for cnt in contours:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 8)
