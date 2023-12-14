# Link to dataset : https://drive.google.com/drive/folders/13Fk6D_sB3CGwTxuS3i8FC9ko49xHDsRj?usp=drive_link

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error reading image at: {file_path}")
        return None

    edges = cv2.Canny(img, 100, 200)
    _, segmented_img = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
    resized_img = cv2.resize(segmented_img, (64, 64))
    normalized_img = resized_img / 255.0

    return normalized_img

def load_dataset(data_dir):
    images = []
    labels = []

    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)

        if os.path.isdir(category_dir):
            for file_name in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file_name)

                if os.path.isfile(file_path):  # Check if it's a file
                    processed_img = preprocess_image(file_path)

                    if processed_img is not None:
                        images.append(processed_img)
                        labels.append(1 if category == 'cancer' else 0)  # Assign labels
                else:
                    print(f"Ignoring non-file item: {file_path}")

    return np.array(images), np.array(labels)

def augment_data(images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        augmented_img = datagen.flow(img).next()[0]
        augmented_images.append(augmented_img)

    return np.array(augmented_images)

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Adding dropout for regularization
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Load and preprocess the dataset
    data_dir = './OralCancer/train'
    images, labels = load_dataset(data_dir)

    # Reshape images for the CNN model
    images = images.reshape((-1, 64, 64, 1))

    # Split the dataset into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Augment the training data
    augmented_train_images = augment_data(train_images)

    # Concatenate original and augmented training data
    final_train_images = np.concatenate([train_images, augmented_train_images])
    final_train_labels = np.concatenate([train_labels, train_labels])  # Keep labels the same for augmented data

    # Build and train the model
    input_shape = (64, 64, 1)
    model = build_model(input_shape)
    model.fit(final_train_images, final_train_labels, epochs=25, validation_data=(test_images, test_labels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

    model.save('alpha_beta_oral_cancer_model.h5')

