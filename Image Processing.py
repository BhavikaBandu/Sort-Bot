import cv2
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

# Loop through fresh banana images
for filename in os.listdir(os.path.join(data_path, fresh_folder)):
    img = cv2.imread(os.path.join(data_path, fresh_folder, filename))
    images.append(img)
    labels.append(0)  # Fresh label

# Loop through rotten banana images
for filename in os.listdir(os.path.join(data_path, rotten_folder)):
    img = cv2.imread(os.path.join(data_path, rotten_folder, filename))
    images.append(img)
    labels.append(1)  # Rotten label

# Loop through raw banana images (uncomment if including raw class)
for filename in os.listdir(os.path.join(data_path, raw_folder)):
    img = cv2.imread(os.path.join(data_path, raw_folder, filename))
    images.append(img)
    labels.append(2)  # Raw label

# Preprocess images (function)
def preprocess(image):
    # Resize the image
    image = cv2.resize(image, (224, 224))

    # Convert to grayscale (optional)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flatten the image into a feature vector
    return image.flatten()

# Preprocess all images
processed_images = [preprocess(img) for img in images]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Function to predict banana from an image file
def predict_from_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Preprocess the image
    processed_image = preprocess(image)

    # Predict banana class
    prediction = clf.predict([processed_image])[0]
    class_text = "Fresh" if prediction == 0 else ("Rotten" if prediction == 1 else ("Raw" if prediction == 2 else "Nothing"))

    # Display image with prediction text
    cv2.putText(image, f"Prediction: {class_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Banana Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "PATH TO TEST IMAGE"
predict_from_image(image_path)
