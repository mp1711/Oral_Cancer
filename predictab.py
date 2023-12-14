# Link to dataset : https://drive.google.com/drive/folders/13Fk6D_sB3CGwTxuS3i8FC9ko49xHDsRj?usp=drive_link

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error reading image at: {file_path}")
        return None

    edges = cv2.Canny(img, 100, 200)
    _, segmented_img = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
    resized_img = cv2.resize(segmented_img, (64, 64))
    normalized_img = resized_img / 255.0

    return np.expand_dims(normalized_img, axis=0)

# Load the min_max_oral_cancer_model.h5 model
min_max_model_path = './models/oral_cancer_min_max.h5'
min_max_model = load_model(min_max_model_path)

# Path to the images
images_folder = './OralCancer/validate'

# Initialize lists to store true labels and predicted probabilities
true_labels = []
predicted_probabilities = []

# Process each image
for class_label in ['cancer', 'non-cancer']:
    class_folder = os.path.join(images_folder, class_label)

    for image_file in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_file)
        input_image = preprocess_image(image_path)

        if input_image is not None:
            # Make predictions using the min_max_model
            segment_probabilities = min_max_model.predict(input_image)[0].tolist()

            # Combine adjacent probabilities iteratively
            MAX = True
            while len(segment_probabilities) > 1:
                new_probabilities = []
                for i in range(0, len(segment_probabilities), 2):
                    if MAX: 
                        if i + 1 < len(segment_probabilities):
                            parent_probability = max(segment_probabilities[i], segment_probabilities[i + 1])
                            new_probabilities.append(parent_probability)
                        else:
                            # For odd length, the last probability remains unchanged
                            new_probabilities.append(segment_probabilities[i])
                        MAX = not MAX
                    else:
                        if i + 1 < len(segment_probabilities):
                            parent_probability = min(segment_probabilities[i], segment_probabilities[i + 1])
                            new_probabilities.append(parent_probability)
                        else:
                            # For odd length, the last probability remains unchanged
                            new_probabilities.append(segment_probabilities[i])
                        MAX = not MAX

                segment_probabilities = new_probabilities

            root_probability = segment_probabilities[0]

            # Assign true label based on the folder (cancer or non-cancer)
            true_label = 1 if class_label == 'cancer' else 0
            true_labels.append(true_label)

            # Store the root probability
            predicted_probabilities.append(root_probability)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
roc_auc = auc(fpr, tpr)

# Print ROC AUC
print(f"ROC AUC: {roc_auc}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
