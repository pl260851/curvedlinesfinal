import cv2
import numpy as np
from image_processing_file import process_image_for_curves
from contour_file import find_and_draw_contours

def draw_roi(img, vertices):
    # Draw the region of interest as a polygon on the image.
    cv2.polylines(img, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

def capture_and_process_video():
    # Initialize video capture from the default camera.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture")
        exit()

    ret, frame = cap.read()
    if ret:
        roi_vertices = define_roi(frame)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame to detect edges within the roi.
            masked_canny = process_image_for_curves(frame, roi_vertices)
            # Find contours in the processed frame and draw them along with the midline.
            find_and_draw_contours(frame, masked_canny)
            # Draw the roi on the frame.
            draw_roi(frame, roi_vertices)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to read frame from camera")

def define_roi(frame):
    #The roi based on the frame size.
    rows, cols, _ = frame.shape
    roi_width = cols * 0.6
    roi_height = rows * 0.6
    top_left_vertex = (int((cols - roi_width) / 2), int((rows - roi_height) / 2))
    bottom_right_vertex = (int((cols + roi_width) / 2), int((rows + roi_height) / 2))

    # Return the vertices of the roi.
    roi_vertices = np.array([
        top_left_vertex,
        (top_left_vertex[0], bottom_right_vertex[1]),
        bottom_right_vertex,
        (bottom_right_vertex[0], top_left_vertex[1])
    ], dtype=np.int32)
    return roi_vertices
