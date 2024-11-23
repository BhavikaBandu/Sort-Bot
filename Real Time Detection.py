import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

# Define data paths (modify these to your actual folder structure)
data_path = "PATH TO THE FOLDER WITH THE 3 DATASETS"
fresh_folder = "fresh_banana"
rotten_folder = "rotten_banana"
raw_folder = "raw_banana"

# Load images and labels
images = []
labels = []

# Function to check if the file is an image
def is_image(file):
    return file.endswith(('.jpg', '.jpeg', '.png'))

# Loop through fresh banana images
for filename in os.listdir(os.path.join(data_path, fresh_folder)):
    if is_image(filename):
        img = cv2.imread(os.path.join(data_path, fresh_folder, filename))
        if img is not None:
            images.append(img)
            labels.append(0)  # Fresh label

# Loop through rotten banana images
for filename in os.listdir(os.path.join(data_path, rotten_folder)):
    if is_image(filename):
        img = cv2.imread(os.path.join(data_path, rotten_folder, filename))
        if img is not None:
            images.append(img)
            labels.append(1)  # Rotten label

# Loop through raw banana images (uncomment if including raw class)
for filename in os.listdir(os.path.join(data_path, raw_folder)):
    if is_image(filename):
        img = cv2.imread(os.path.join(data_path, raw_folder, filename))
        if img is not None:
            images.append(img)
            labels.append(2)  # Raw label

# Preprocess images (function)
def preprocess(image):
    # Resize the image to a fixed size
    image = cv2.resize(image, (224, 224))

    # Normalize the pixel values (0-255 to 0-1)
    normalized_image = gray / 255.0

    # Flatten the image into a feature vector
    return normalized_image.flatten()

# Preprocess all images
processed_images = [preprocess(img) for img in images]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Function to predict banana from an image file
def predict_from_image(image):
    # Preprocess the image
    processed_image = preprocess(image)

    # Predict banana class
    prediction = clf.predict([processed_image])[0]
    class_text = "Fresh" if prediction == 0 else ("Rotten" if prediction == 1 else ("Raw" if prediction == 2 else "Nothing"))

    # Return the prediction text
    return class_text

# Real-time image capture function
def real_time_capture():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)  # '0' is typically the default camera on most systems
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the frame
        class_text = predict_from_image(frame)  # Get prediction for the current frame
        
        # Display prediction text on the frame
        cv2.putText(frame, f"Prediction: {class_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the current frame
        cv2.imshow("Banana Classification - Real Time", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time image capture
real_time_capture()
