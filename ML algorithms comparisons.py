#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(image_path):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def detect_and_draw_potholes(image_path, classifier):
    image = cv2.imread(image_path)
    features = extract_features(image_path)

    # Predict if the image contains a pothole
    prediction = classifier.predict([features])

    # If a pothole is detected, draw a bounding box
    if prediction[0] == 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 10:  # Adjust the threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                print(f"Detected pothole with area: {contour_area}")

    # Display the image with bounding boxes using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

image_folder = "E://python//New folder//dataset 10"

image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png", ".jpg", ".jpg"))]

# Initialize classifiers
classifiers = [
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("k-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Support Vector Machine", SVC(kernel='linear', C=1))
]

# Initialize lists to store features and labels
X = []  # Features
y = []  # Labels (1 for pothole, 0 for non-pothole)

# Process each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_file)

    # Extract features from the image
    features = extract_features(image_path)

    # Add features to X
    X.append(features)

    # Determine the label (1 for pothole, 0 for non-pothole)
    label = 1 if "pothole" in image_file.lower() else 0
    y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


results = {
    "Algorithm": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []  # Add F1 Score
}


for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)  # Calculate F1 Score
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    results["Algorithm"].append(clf_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)  # Add F1 Score

    print(f"Classifier: {clf_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")  # Print F1 Score
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 50)

# Visualize line graphs for Algorithm vs. Accuracy, Algorithm vs. Precision, Algorithm vs. Recall, Algorithm vs. F1 Score
plt.figure(figsize=(16, 6))
plt.subplot(1, 4, 1)
sns.lineplot(x="Algorithm", y="Accuracy", data=results)
plt.xticks(rotation=45)
plt.title("Algorithm vs. Accuracy")

plt.subplot(1, 4, 2)
sns.lineplot(x="Algorithm", y="Precision", data=results)
plt.xticks(rotation=45)
plt.title("Algorithm vs. Precision")

plt.subplot(1, 4, 3)
sns.lineplot(x="Algorithm", y="Recall", data=results)
plt.xticks(rotation=45)
plt.title("Algorithm vs. Recall")

plt.subplot(1, 4, 4)
sns.lineplot(x="Algorithm", y="F1 Score", data=results)
plt.xticks(rotation=45)
plt.title("Algorithm vs. F1 Score")

plt.tight_layout()
plt.show()


# In[ ]:




