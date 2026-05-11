import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Load and preprocess dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    # Update these categories based on your folder names
    categories = ['Tomato_healthy', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Target_Spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato_Septoria_leaf_spot']

    for idx, category in enumerate(categories):
        category_folder = os.path.join(folder, category)
        if not os.path.exists(category_folder):
            print(f"Warning: Skipping category '{category}' - folder not found.")
            continue
        for filename in os.listdir(category_folder):
            img = cv2.imread(os.path.join(category_folder, filename))
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
                images.append(img)
                labels.append(idx)  # Assign label based on category
    return np.array(images), np.array(labels)


# Feature extraction using Histogram of Oriented Gradients (HOG)
def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Convert to uint8 and grayscale before applying HOG
        img_uint8 = (img * 255).astype(np.uint8)  # Convert float64 to uint8
        gray_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        feature = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
        hog_features.append(feature)
    return np.array(hog_features)


# Plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Main script
if __name__ == "__main__":
    # Set the path to your dataset
    data_folder = 'test'  # Replace with actual path to dataset

    # Step 1: Load and preprocess the images
    images, labels = load_images_from_folder(data_folder)

    # Check if we have loaded any images
    if len(images) == 0:
        print("No images were loaded. Please check the folder structure and paths.")
        exit(1)

    images = images / 255.0  # Normalize pixel values to [0, 1]

    # Step 2: Extract HOG features
    features = extract_hog_features(images)

    # Step 3: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Step 4: Train and evaluate Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f'Decision Tree Accuracy: {accuracy_dt * 100:.2f}%')
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print('Confusion Matrix - Decision Tree:\n', cm_dt)

    # Plot Decision Tree confusion matrix
    plot_confusion_matrix(cm_dt, "Confusion Matrix - Decision Tree")

    # Step 5: Train and evaluate Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f'Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%')
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    print('Confusion Matrix - Naive Bayes:\n', cm_nb)

    # Plot Naive Bayes confusion matrix
    plot_confusion_matrix(cm_nb, "Confusion Matrix - Naive Bayes")

    # Final accuracy comparison
    print(f"Final Comparison:\nDecision Tree Accuracy: {accuracy_dt * 100:.2f}%\nNaive Bayes Accuracy: {accuracy_nb * 100:.2f}%")
