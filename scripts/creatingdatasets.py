import os
import shutil
import random

# Define the paths for training, validation, and testing data
base_dir = "../datasets"  # Replace with the path to your dataset folder
training_dir = os.path.join(base_dir, "training_data")
validation_dir = os.path.join(base_dir, "validation_data")
testing_dir = os.path.join(base_dir, "testing_data")

# Iterate over each letter folder in the training data
for letter in os.listdir(training_dir):
    letter_dir = os.path.join(training_dir, letter)

    # Make sure it's a directory
    if not os.path.isdir(letter_dir):
        continue

    # Create corresponding letter folders in validation and testing directories
    os.makedirs(os.path.join(validation_dir, letter), exist_ok=True)
    os.makedirs(os.path.join(testing_dir, letter), exist_ok=True)

    # List all images in the letter folder
    images = os.listdir(letter_dir)
    random.shuffle(images)  # Shuffle to randomly select images

    # Move 500 images to validation and 500 images to testing
    for i in range(500):
        shutil.move(os.path.join(letter_dir, images[i]), os.path.join(validation_dir, letter))
    for i in range(500, 1000):
        shutil.move(os.path.join(letter_dir, images[i]), os.path.join(testing_dir, letter))

print("Images moved successfully.")