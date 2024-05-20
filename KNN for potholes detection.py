#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to extract features from an image (average pixel intensity)
def extract_features(image):
    return np.mean(image)

# Load the image
image = cv2.imread('E:\\python\\New folder\\new image 1/21.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (1, 1), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 90, 150)

# Find contours in the edges image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over all contours and filter out small ones
filtered_contours = []
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Adjust the threshold as needed
        filtered_contours.append(contour)

# Extract features from the filtered contours
features = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y+h, x:x+w]
    features.append(extract_features(roi))

# Create labels for the features (1 for pothole, 0 for non-pothole)
labels = np.ones(len(features))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape the features arrays
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Draw contours on the original image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Pothole Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

