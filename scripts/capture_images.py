import cv2
import os
import string


def capture_images(cap, dataset_type, class_name, num_images, delay, start_x, start_y, end_x, end_y):
    base_dir = os.path.join(os.getcwd(), '../datasets')
    target_dir = os.path.join(base_dir, dataset_type, class_name.upper())
    os.makedirs(target_dir, exist_ok=True)

    existing_files = os.listdir(target_dir)
    count = max([get_image_count(f, class_name) for f in existing_files] + [0])
    start_count = count + 1

    capturing = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read from the camera.")
            return

        roi = frame[start_y:end_y, start_x:end_x].copy()
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        if capturing:
            instruction_text1 = f'Capturing for {class_name}...'
            img_name = f"{class_name.upper()}_{count + 1}.png"
            img_path = os.path.join(target_dir, img_name)
            cv2.imwrite(img_path, roi)
            print(f"{img_name} written!")
            count += 1
            cv2.waitKey(delay)
        else:
            instruction_text1 = f'Press "c" to start capturing for letter {class_name}.'

        instruction_text2 = f'Captured {count - start_count + 1}/{num_images} for {class_name}.'
        cv2.putText(frame, instruction_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, instruction_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Capture Images', frame)

        k = cv2.waitKey(1)
        if k == ord('c') and not capturing:
            capturing = True

        if (count - start_count + 1) >= num_images:
            break

def get_image_count(filename, class_name):
    try:
        if filename.startswith(class_name.upper() + '_'):
            return int(filename.split('_')[-1].split('.')[0])
        return 0
    except ValueError:
        return 0


if __name__ == "__main__":
    dataset_type_dict = {
        1: "training_data",
        2: "testing_data",
        3: "validation_data"
    }

    dataset_type_input = 0
    while dataset_type_input not in dataset_type_dict:
        try:
            dataset_type_input = int(
                input("Choose the dataset type: \n1 for training_data\n2 for testing_data\n3 for validation_data\n")
            )
        except ValueError:
            print("Please enter a valid number (1, 2, or 3).")
    dataset_type = dataset_type_dict[dataset_type_input]

    num_images = int(input("Enter the number of images you want to capture for each letter: "))
    delay = int(input("Enter the delay between automatic captures (in milliseconds): "))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read from the camera.")
        cap.release()
        exit()

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    start_x, start_y = center_x - 128, center_y - 128
    end_x, end_y = center_x + 128, center_y + 128

    # Define custom letter sequence
    letter_sequence = list(string.ascii_uppercase)
    letter_sequence.remove('H')  # Remove H
    letter_sequence.extend(['AE', 'OE'])  # Add AE and OE

    for letter in letter_sequence:
        capture_images(cap, dataset_type, letter, num_images, delay, start_x, start_y, end_x, end_y)

    cap.release()
    cv2.destroyAllWindows()