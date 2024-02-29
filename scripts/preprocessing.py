import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(img_path: Path, output_dir: Path) -> np.array:
    # Read the image in color mode to convert to HSV later
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Image not found at {img_path}")

    # Resize the image to 128x128 pixels
    img = cv2.resize(img, (224, 224))

    img = img / 255.0

    # ---- Skin Masking ----
    # Convert the resized image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the resized image to HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV bounds for skin masking
    # lower_bound = np.array([0, 40, 30], dtype="uint8")
    # upper_bound = np.array([43, 255, 254], dtype="uint8")

    # Create a binary mask where pixel values within the bounds are set to 1 and others to 0
    # skin_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Perform a bitwise_and operation with the grayscale image and the skin mask to retain only the skin regions in the image
    # skin = cv2.bitwise_and(gray, gray, mask=skin_mask)

    # ---- Thresholding ----
    # Apply thresholding
    # _, thresh = cv2.threshold(skin, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ---- Canny Edge Detection ----
    # Apply Canny Edge Detection
    # edges = cv2.Canny(skin_mask, 100, 200)

    # Save the preprocessed image to the corresponding output directory
    cv2.imwrite(str(output_dir / img_path.name), img)


def apply_augmentation(img: np.array, output_dir: Path, img_file: Path):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    img = img.reshape((1,) + img.shape + (1,))  # Reshape the image to (1, 128, 128, 1)
    for i, batch in enumerate(datagen.flow(img, batch_size=1)):
        augmented_img = batch.squeeze()
        augmented_img_path = output_dir / f"{img_file.stem}_aug_{i}{img_file.suffix}"
        cv2.imwrite(str(augmented_img_path), augmented_img * 255)
        if i == 4:  # We want to create 5 augmented images including the original
            break


def process_dataset(dataset_type: str):
    input_dir = base_input_dir / dataset_type
    output_dir = base_output_dir / dataset_type
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            output_class_dir = output_dir / class_dir.name
            output_class_dir.mkdir(exist_ok=True)

            for img_file in class_dir.iterdir():
                if img_file.suffix in ['.jpg', '.jpeg', '.png']:
                    preprocess_image(img_file, output_class_dir)

                    # Uncomment the next lines to apply augmentation on the training data
                    # if dataset_type == 'training_data':
                    # apply_augmentation(img, output_class_dir, img_file)


base_input_dir = Path('../datasets')
base_output_dir = Path('processed_datasets')

datasets = ['training_data', 'testing_data', 'validation_data']

for dataset in datasets:
    process_dataset(dataset)

print("Preprocessing and Augmentation Completed!")
