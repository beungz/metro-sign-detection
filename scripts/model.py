import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import shutil
from scripts.build_features import extract_hog_feature
from collections import Counter



# NN-based Deep Learning Model: YOLOv8

def train_yolo(epochs=200, img_size=960, batchsize=16, device='cuda'):
    '''Train a YOLOv8 model via transfer learning and return the trained model.'''

    # Load the pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Transfer learning with training dataset
    # Augmentations are automatically applied by YOLOv8 to training
    model.train(
        data="data/outputs/data.yaml",  # Path of dataset YAML file
        epochs=epochs,                  # Number of training epochs
        imgsz=img_size,                 # Image size
        batch=batchsize,                # Batch size
        device=device,                  # GPU or CPU
        augment=True,                   # Enable data augmentation
        save=True,                      # Save the model after training
        project="models/deep_learning", # Main directory for saving the model
        name="training_results"         # Directory name for saving the results
    )

    # Copy the best model from its original path to the final path
    best_model_original_path = os.path.join("models", "deep_learning", "training_results", "weights", "best.pt")
    best_model_final_path = os.path.join("models", "deep_learning", "best.pt")
    shutil.copyfile(best_model_original_path, best_model_final_path)

    # Validate the model on the test dataset
    metrics = model.val(
        data="data/outputs/data.yaml",  # Path of dataset YAML file
        split='test',                   # Use the test split for final evaluation
        imgsz=img_size,                 # Image size
        device=device,                  # GPU or CPU
        project="models/deep_learning", # Main directory for saving the model
        name="test_results"             # Directory name for saving the results
    )

    # Print metrics
    print(metrics)

    # Compute y_pred and y_true for sklearn-style metrics
    y_test = []
    y_pred = []

    test_images_dir = os.path.join("data", "outputs", "test", "images")
    test_labels_dir = os.path.join("data", "outputs", "test", "labels")

    for image_name in os.listdir(test_images_dir):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Image path and label path
        img_path = os.path.join(test_images_dir, image_name)
        label_path = os.path.join(test_labels_dir, os.path.splitext(image_name)[0] + ".txt")

        # Get ground truth labels
        with open(label_path, "r") as f:
            for line in f:
                cls_id = int(line.split()[0])
                y_test.append(cls_id)

        # Run inference
        results = model.predict(img_path, imgsz=img_size)
        boxes = results[0].boxes

        # Take top-1 prediction per ground-truth box
        if boxes is not None and len(boxes.cls) > 0:
            pred_classes = boxes.cls.cpu().numpy().astype(int)
            pred_count = len(pred_classes)
            # Append predicted class
            y_pred.extend(pred_classes[:len(open(label_path).readlines())])
        else:
            # No prediction — use -1 or dummy class (optional)
            y_pred.extend([-1] * len(open(label_path).readlines()))

    # y_test = np.array(y_test)
    # y_pred = np.array(y_pred[:len(y_test)])
    min_len = min(len(y_test), len(y_pred))
    y_test = np.array(y_test[:min_len])
    y_pred = np.array(y_pred[:min_len])

    # Evaluate model
    test_accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy: {:.6f}".format(test_accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, metrics



def predict_yolo_single_image(model, image):
    '''Predict metro station name(s) and bounding box(es) around station signage(s) in the input image using the trained YOLOv8 model.'''

    # Get the model's predictions and save the resulting image with bounding boxes
    results = model.predict(source=image, save=True, project="data", name="demo_predictions")
    
    # Extract all predicted class_ids (station names) from the results
    result = results[0]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # Map class_ids to station names, then remove duplicates
    station_names = [model.names[class_id] for class_id in class_ids]
    station_names_unique = list(set(station_names))
    
    # Rename the saved image (default name = image0.jpg) with a timestamp
    result_dir = results[0].save_dir
    original_path = os.path.join(result_dir, "image0.jpg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_image_name = f"detected_{timestamp}.jpg"
    new_resulting_image_path = os.path.join(result_dir, new_image_name)
    os.rename(original_path, new_resulting_image_path)

    return station_names_unique, new_resulting_image_path



# Classical Machine Learning Model: Support Vector Classifier (SVC)

def train_svc(X_train_full_scaled, y_train_full, X_test_scaled, y_test, param_grid):
    '''Train a Support Vector Classifier (SVC) model with hyperparameter tuning using GridSearchCV and evaluate its performance.'''

    # Initiate SVC model
    model = SVC(probability=True)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1, return_train_score=True)

    # Fit the model to the training set
    grid_search.fit(X_train_full_scaled, y_train_full)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    cv_accuracy = grid_search.best_score_
    print("Best Cross-validation Accuracy: {:.6f}".format(cv_accuracy))

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    # Predict class labels for test set
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate model
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.6f}".format(test_accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the best model
    best_model_final_path = os.path.join("models", "classical_machine_learning", "best_svc_model.pkl")
    joblib.dump(best_model, best_model_final_path)

    return best_model, cv_accuracy, test_accuracy



def predict_svc_single_image(model, cropped_signage_image):
    '''Predict metro station name in the cropped signage image using the trained SVC model.'''
    # Extract HOG features from the cropped signage image
    X_before_scaled = extract_hog_feature(cropped_signage_image, target_size=(128, 64))

    # Load the scaler
    scaler_path = os.path.join("models", "classical_machine_learning", "hog_scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Scale the features
    X_scaled = scaler.transform(X_before_scaled.reshape(1, -1))

    # Predict station name
    y_pred = model.predict(X_scaled)
    station_name = y_pred[0] if len(y_pred) == 1 else y_pred.tolist()
    
    return station_name



# Naive Model

def predict_naive_evaluate_test_set(options, y_train_full, y_test):
    '''Predict metro station name with a naive approach based on the input options.
        options:1 always predicts 'Siam', which has the highest ridership, according to the statistics from BTS, the metro operator. 
        options:2 randomly predicts one of the 8 stations, based on the frequency of the stations in the training set.'''

    if options == 1:
        # Always predict 'Siam' as it has the highest ridership, according to the statistics from BTS, the metro operator. 
        y_pred = ['Siam'] * len(y_test)

        # Evaluate the naive model
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy: {:.6f}".format(test_accuracy))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    elif options == 2:
        # Calcuate the frequency distribution of stations in the training set

        class_counter = Counter(y_train_full)
        total = sum(class_counter.values())
        classes = list(class_counter.keys())
        probabilities = [class_counter[c] / total for c in classes]

        print("Station Frequency Distribution:\n")
        for cls, prob in zip(classes, probabilities):
            print(f"  {cls}: {prob*100:.2f}%")

        # Predict by randomly choosing a station based on the frequency distribution
        np.random.seed(42)
        y_pred_random = np.random.choice(classes, size=len(y_test), p=probabilities)

        # Evaluate the naive model
        test_accuracy = accuracy_score(y_test, y_pred_random)
        print("\nTest Accuracy: {:.6f}".format(test_accuracy))
        print("Classification Report:\n", classification_report(y_test, y_pred_random))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_random))

    return test_accuracy
