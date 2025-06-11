from scripts.make_dataset import remove_corrupted_duplicates_and_rename_images, split_dataset, crop_augment_images_for_svc, split_dataset_for_svc
from scripts.build_features import build_features_svc
from scripts.model import train_yolo, train_svc, predict_naive_evaluate_test_set



def main():
    '''Main function to run the entire pipeline for detecting Bangkok Metro Station Signage.'''
    # A. Get Dataset
    print("A. Get Dataset")

    # Step A1: Gather the raw dataset from Google Maps
    # Gather images with metro station signages from Google Maps/Photos (approx 100 photos per station) and save them in the data/raw folder.


    # List of station folder names corresponding to the 8 stations in the BTS Silom Line (Dark Green Line without extension)
    station_folder_name = [
        "Siam",
        "National_Stadium", 
        "Ratchadamri", 
        "Sala_Daeng", 
        "Chong_Nonsi", 
        "Saint_Louis", 
        "Surasak", 
        "Saphan_Taksin"
    ]


    # Step A2: Remove corrupted images, duplicates, and rename images, then save them in the data/processed_before_roboflow folder
    # This step was done before uploading to Roboflow, so it should be commented out if the annotated dataset exported from Roboflow is ready for the next step.
    print("Step A2: Remove corrupted images, duplicates, and rename images.")
    remove_corrupted_duplicates_and_rename_images(station_folder_name)


    # Step A3: Upload the dataset to Roboflow, annotate the images, draw bounding boxes manually, and then export the dataset in YOLO format.
    # Put the exported dataset in the data/processed folder.


    # Step A4: Split the dataset into training, validation, and test sets, then save them in the data/outputs folder.
    print("Step A4: Split the dataset into training, validation, and test sets.")
    split_dataset(station_folder_name)


    # List of labels/folders corresponding to YOLO class IDs (folder names have underscores while YOLO class IDs do not)
    label_map = {
        0: 'Chong_Nonsi', 
        1: 'National_Stadium',
        2: 'Ratchadamri', 
        3: 'Saint_Louis', 
        4: 'Sala_Daeng', 
        5: 'Saphan_Taksin', 
        6: 'Siam', 
        7: 'Surasak'
        }


    # Step A5: Crop the images in the training, validation, and test sets to create a new dataset with only the metro station signage. This is for the SVC model.
    print("Step A5: Crop the images for SVC model.")
    crop_augment_images_for_svc(label_map)


    # Step A6: Build features and labels for SVC model
    print("Step A6: Build features and labels for SVC model.")
    features_by_split, labels_by_split = build_features_svc()


    # Step A7: Prepare the training and test data for SVC model
    print("Step A7: Prepare the training and test data for SVC model.")
    X_train_full_scaled, y_train_full, X_test_scaled, y_test = split_dataset_for_svc(features_by_split, labels_by_split)


    # B. Train YOLOv8 and SVC models
    print("B. Train YOLOv8 and SVC models")

    # Step B1: Train YOLOv8 model for detecting Bangkok Metro Station Signage
    print("Step B1: Train YOLOv8 model for detecting Bangkok Metro Station Signage")
    best_yolo_model, yolo_metrics = train_yolo(epochs=200, img_size=960, batchsize=16, device='cuda')


    # Step B2: Train SVC model for detecting Bangkok Metro Station Signage
    print("Step B2: Train SVC model for detecting Bangkok Metro Station Signage")

    # Define the list of hyperparameter for cross validation
    param_grid = {
        'kernel': ['rbf'],  # Type of kernel
        'C': [0.1, 1, 10],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’
        'tol': [1e-3, 1e-4]  # Tolerance for stopping criteria
    }

    best_svc_model, svc_cv_accuracy, svc_test_accuracy = train_svc(X_train_full_scaled, y_train_full, X_test_scaled, y_test, param_grid)

    # Step B3: Evaluate naive model with option 1 
    # Always predicts 'Siam', which has the highest ridership, according to the statistics from BTS, the metro operator.
    print("Step B3: Evaluate naive model with option 1")
    naive_1_accuracy = predict_naive_evaluate_test_set(1, y_train_full, y_test)

    # Step B4: Evaluate naive model with option 2
    # Randomly predicts one of the 8 stations, based on the frequency of the stations in the training set.
    print("Step B4: Evaluate naive model with option 2")
    naive_2_accuracy = predict_naive_evaluate_test_set(2, y_train_full, y_test)



if __name__ == "__main__":
    main()
