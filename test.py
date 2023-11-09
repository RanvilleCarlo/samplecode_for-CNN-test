
import cv2
import numpy as np
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the dataset of images
dataset_path = "/path/to/dataset"
images = []
for i in range(10):
    image_path = dataset_path + "/image{}.jpg".format(i)
    image = cv2.imread(image_path)
    images.append(image)

# Preprocess the images to prepare them for damage detection
processed_images = []
for image in images:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    processed_images.append(thresh)

# Use a damage detection algorithm to identify the damages in the images
damage_images = []
for image in processed_images:
    # Apply morphological transformations to fill gaps and remove noise
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Find contours in the image
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Classify the damages based on severity
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            severity = "minor"
        elif area < 1000:
            severity = "moderate"
        else:
            severity = "major"
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, severity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    damage_images.append(image)

# Save the images with the bounding boxes and severity labels drawn on them
for i, image in enumerate(damage_images):
    cv2.imwrite(dataset_path + "/damage_image{}.jpg".format(i), image)
    import matplotlib.pyplot as plt

    # Load the dataset of wall images
    dataset_path = "/path/to/dataset"
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    # Preprocess the images to prepare them for the CNN model
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical')

    # Define the CNN model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(),
                  metrics=['accuracy'])

    # Train the CNN model on the training set
    history = model.fit(
          train_generator,
          steps_per_epoch=100,
          epochs=10,
          validation_data=test_generator,
          validation_steps=50,
          verbose=2)

    # Evaluate the CNN model on the testing set
    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Use the trained CNN model to classify the damages in the wall images
    damage_images = []
    for i in range(10):
        image_path = dataset_path + "/image{}.jpg".format(i)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))
        image = np.reshape(image, [1, 150, 150, 3])
        prediction = model.predict(image)
        if np.argmax(prediction) == 0:
            severity = "minor"
        elif np.argmax(prediction) == 1:
            severity = "moderate"
        else:
            severity = "major"
        cv2.putText(image, severity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        damage_images.append(image)

    # Save the images with the bounding boxes and severity labels drawn on them
    for i, image in enumerate(damage_images):
        cv2.imwrite(dataset_path + "/damage_image{}.jpg".format(i), image)
