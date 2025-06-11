import os
import shutil
from PIL import Image
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
from imgaug import augmenters as iaa



def is_image_corrupted(filepath):
    '''Check if an image file is corrupted.'''
    try:
        with Image.open(filepath) as img:
            # Verify image integrity
            img.verify()
        return False
    except Exception:
        return True



def get_file_hash(filepath):
    '''Return SHA256 hash of file content.'''
    hash_func = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()



def remove_corrupted_duplicates_and_rename_images(station_folder_name):
    '''Remove corrupted images, duplicates, and rename images in the dataset.'''

    seen_hashes = set()
    copied_count = 0

    for station in station_folder_name:
        seen_hashes = set()
        image_id = 0

        station_path = os.path.join("data", "raw", station)
        if not os.path.isdir(station_path):
            continue
        dst_station_path = os.path.join("data", "processed_before_roboflow", station)
        os.makedirs(dst_station_path, exist_ok=True)
        for fname in os.listdir(station_path):
            src_img_path = os.path.join(station_path, fname)
            if not os.path.isfile(src_img_path):
                continue
            # Check for corrupted files
            if is_image_corrupted(src_img_path):
                print(f"Corrupted: {src_img_path}")
                continue
            # Check for duplicates
            file_hash = get_file_hash(src_img_path)
            if file_hash in seen_hashes:
                print(f"Duplicate: {src_img_path}")
                continue
            seen_hashes.add(file_hash)
            # Copy to destination folder
            image_id += 1
            ext = os.path.splitext(fname)[1]
            new_fname = f"{station}_{image_id:03d}{ext}"
            dst_img_path = os.path.join(dst_station_path, new_fname)
            shutil.copy2(src_img_path, dst_img_path)
            copied_count += 1

    print(f"{copied_count} unique, valid images are processed and copied to data/processed_before_roboflow folder.")
    return



def split_dataset(station_folder_name):
    '''Split the dataset into training, validation, and test sets, from the dataset that is in YOLO format with images and labels.'''

    input_base_dir = os.path.join("data", "processed")
    images_dir = os.path.join(input_base_dir, "train", "images")
    labels_dir = os.path.join(input_base_dir, "train", "labels")
    output_base_dir = os.path.join("data", "outputs")

    # Set the split ratio for train, validation, and test sets
    split_ratio = (0.8, 0.1, 0.1)  # train, valid, test

    # Load image-label pairs and classes
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
    samples = []
    classes_per_image = []

    # Iterate through image files and their corresponding label files
    for img_file in image_files:
        label_file = img_file.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path):
            continue
        # Read the label file to get class IDs
        with open(label_path, "r") as f:
            class_ids = [line.strip().split()[0] for line in f if line.strip()]
        if class_ids:
            samples.append((img_file, label_file))
            classes_per_image.append(class_ids[0])  # Use first class ID (or a representative one)

    # Split data stratified by class
    train_val_files, test_files = train_test_split(samples, test_size=split_ratio[2], stratify=classes_per_image, random_state=42)
    train_val_classes = [label[0] for _, label in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]), stratify=train_val_classes, random_state=42)

    splits = {
        "train": train_files,
        "valid": val_files,
        "test": test_files
    }

    # Copy files to split folders
    for split_name, files in splits.items():
        # Create directories for images and labels in the output base directory
        img_out = os.path.join(output_base_dir, split_name, "images")
        lbl_out = os.path.join(output_base_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        # Copy images and labels to the respective split directories
        for img_file, label_file in files:
            shutil.copy(os.path.join(images_dir, img_file), os.path.join(img_out, img_file))
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(lbl_out, label_file))

    for fname in os.listdir(input_base_dir):
        # Copy any additional files (including data.yaml) to the output base directory
        input_path = os.path.join(input_base_dir, fname)
        output_path = os.path.join(output_base_dir, fname)
        if os.path.isfile(input_path):
            shutil.copy2(input_path, output_path)

    print("Stratified split complete.")

    return



def convert_yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    '''Convert YOLO format to image crops'''
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return max(x1, 0), max(y1, 0), min(x2, img_w - 1), min(y2, img_h - 1)



def crop_augment_images_for_svc(label_map):
    '''Crop images in the training, validation, and test sets for SVC model. This uses YOLO format annotations to crop and get images of metro station signage.'''

    # Minimum input size before resize (must be at least HOG-safe)
    min_width, min_height = 16, 16

    # Define augmentation pipeline
    augmenter = iaa.Sequential([
        iaa.Affine(
            rotate=(-10, 10),
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),
        iaa.Multiply((0.8, 1.2)),  # brightness
        iaa.LinearContrast((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    # Crop images for each split
    splits = ["train", "valid", "test"]
    for split in splits:
        split_dir = os.path.join("data", "outputs", split)
        image_dir = os.path.join(split_dir, "images")
        label_dir = os.path.join(split_dir, "labels")
        output_crop_dir = os.path.join(split_dir, "crops")

        # Create label folders
        for label in label_map.values():
            os.makedirs(os.path.join(output_crop_dir, label), exist_ok=True)

        # Count skipped images per label
        skipped_counts = {label: 0 for label in label_map.values()}
        total_counts = {label: 0 for label in label_map.values()}

        crop_count = 0
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(".jpg"):
                continue
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

            image = cv2.imread(img_path)
            if image is None or not os.path.exists(label_path):
                continue
            
            # Read image dimensions
            img_h, img_w = image.shape[:2]

            # Process each label file
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    x1, y1, x2, y2 = convert_yolo_to_bbox(x, y, w, h, img_w, img_h)
                    crop = image[y1:y2, x1:x2]
                    label = label_map[int(class_id)]
                    total_counts[label] += 1
                    ch, cw = crop.shape[:2]
                    if ch < min_height or cw < min_width:
                        skipped_counts[label] += 1
                        continue
                    cropped_image_name = f"crop_{crop_count}.jpg"
                    crop_path = os.path.join(output_crop_dir, label, cropped_image_name)
                    cv2.imwrite(crop_path, crop)
                    crop_count += 1
                    
                    # Generate 5 augmented versions of cropped images
                    for i in range(5):
                        augmented = augmenter(image=crop)
                        save_name = f"{os.path.splitext(cropped_image_name)[0]}_aug{i}.jpg"
                        cv2.imwrite(os.path.join(output_crop_dir, label, save_name), augmented)

        # Print summary for the split
        print(f"[{split}] Extracted {crop_count} cropped signage images.")
        print(f"[{split}] Skipped images (too small):")

        for label in label_map.values():
            total = total_counts[label]
            skipped = skipped_counts[label]
            print(f"  {label}: {skipped} skipped")
        
    return



def split_dataset_for_svc(features_by_split, labels_by_split):
    '''Split the dataset into training, validation, and test sets for SVC model.'''
    
    X_train = features_by_split["train"]
    y_train = labels_by_split["train"]
    X_valid = features_by_split["valid"]
    y_valid = labels_by_split["valid"]
    X_test = features_by_split["test"]
    y_test = labels_by_split["test"]

    # Combine training and validation sets for cross-validation
    X_train_full = X_train + X_valid
    y_train_full = y_train + y_valid

    # Scale X_train and X_test
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = os.path.join("models", "classical_machine_learning", "hog_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    return X_train_full_scaled, y_train_full, X_test_scaled, y_test
    