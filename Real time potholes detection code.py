#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Define the parameters for pothole detection
MIN_AREA = 1000  # Minimum area of a pothole in pixels
MAX_DEPTH = 10  # Maximum depth of a pothole in cm
THRESHOLD = 0.5  # Threshold for binary segmentation

# Load the video stream from the mobile camera (front camera)
cap = cv2.VideoCapture(0)

# Initialize variables for calculating average depth
depth_values = []
depth_samples = 5  # Number of depth samples to average over

# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to segment the image
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find the contours of the segmented image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop through the contours
    for cnt in contours:
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)

        # Check if the area is above the minimum threshold
        if area > MIN_AREA:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

            # Check if the polygon has four vertices (i.e. it is a rectangle)
            if len(approx) == 4:
                # Draw a bounding box around the contour
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate the depth of the pothole using a simple formula
                depth = MAX_DEPTH * (1 - area / (w * h))
                depth_values.append(depth)

                # Display the depth on the frame
                cv2.putText(
                    frame,
                    f"Depth: {depth:.1f} cm",
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

    # Average the depth values over the last few samples
    if len(depth_values) > depth_samples:
        avg_depth = sum(depth_values[-depth_samples:]) / depth_samples
        cv2.putText(
            frame,
            f"Avg. Depth: {avg_depth:.1f} cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Display the frame with potholes detected
    cv2.imshow("Pothole Detection", frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, exit the loop
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

