#!/usr/bin/env python
# coding: utf-8

import subprocess
from ultralytics import YOLO
import optuna  # For hyperparameter tuning

PathDataSet="/scratch/ritali.ai.iitmandi/temp/data.yaml"
PathModel="/scratch/ritali.ai.iitmandi/runs/detect/train11/weights/best.pt"

# Function to install YOLOv8
def install_yolov8():
    print("Installing YOLOv8 module...")
    # Uncomment the next line if YOLOv8 needs to be installed/reinstalled
    # subprocess.run(['pip', 'install', '--upgrade', 'ultralytics', 'optuna'], check=True)

# Build and train the YOLOv8 model with hyperparameter tuning
def hyperparameter_tuning():
    print("Starting hyperparameter tuning...")

    def objective(trial):
        # Suggest hyperparameters for tuning
        epochs = trial.suggest_int("epochs", 400, 500)
        batch = trial.suggest_categorical("batch", [8, 16, 32])
        imgsz = trial.suggest_categorical("imgsz", [512, 640])
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.95)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # Initialize the model with pre-trained weights
        model = YOLO(PathModel)

        # Train the model with the suggested hyperparameters
        results = model.train(
            data=PathDataSet , # Path to the new dataset
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            optimizer=optimizer,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            patience=10,  # Early stopping patience
            verbose=False  # Suppress detailed logs during tuning
        )

        # Use validation mAP50-95 as the optimization target
        return results["metrics/mAP50-95"]

    # Run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    return study.best_params

# # Train the YOLOv8 model with optimized hyperparameters
# def build_and_train_model(best_params):
#     print("Retraining YOLOv8 model with pre-trained weights and optimized hyperparameters...")

#     # Initialize the model with pre-trained weights
#     model = YOLO(PathModel)

#     # Retrain the model on the new dataset
#     model.train(
#         data=PathDataSet,  # Path to the new dataset
#         epochs=best_params["epochs"],
#         batch=best_params["batch"],
#         imgsz=best_params["imgsz"],
#         optimizer=best_params["optimizer"],
#         lr0=best_params["lr0"],
#         momentum=best_params["momentum"],
#         weight_decay=best_params["weight_decay"],
#         patience=10,  # Early stopping patience
#         verbose=True
#     )


# Train the YOLOv8 model with optimized hyperparameters and save metrics and configuration
def build_and_train_model(best_params):
    print("Retraining YOLOv8 model with pre-trained weights and optimized hyperparameters...")

    # Initialize the model with pre-trained weights
    model = YOLO(PathModel)

    # Define the file to save metrics and hyperparameters
    metrics_file = "training_metrics_and_config.txt"

    # Write hyperparameters and other important information to the file
    with open(metrics_file, "w") as f:
        f.write("YOLOv8 Model Training Log\n")
        f.write("=========================\n")
        f.write("Model Path: {}\n".format(PathModel))
        f.write("Dataset Path: {}\n".format(PathDataSet))
        f.write("Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        f.write("Epoch\tTrain_Loss\tVal_Loss\tmAP50\tmAP50-95\n")  # Header for metrics

    # Callback to log metrics after each epoch
    def log_metrics(epoch, metrics):
        train_loss = metrics["train/loss"]  # Training loss
        val_loss = metrics["val/loss"]  # Validation loss
        map50 = metrics["metrics/mAP50(B)"]  # mAP@0.5
        map50_95 = metrics["metrics/mAP50-95"]  # mAP@0.5:0.95

        # Log metrics to the console
        print(f"Epoch {epoch + 1}: Train Loss={train_loss}, Val Loss={val_loss}, mAP@0.5={map50}, mAP@0.5:0.95={map50_95}")

        # Save metrics to the file
        with open(metrics_file, "a") as f:
            f.write(f"{epoch + 1}\t{train_loss}\t{val_loss}\t{map50}\t{map50_95}\n")

    # Retrain the model on the new dataset
    model.train(
        data=PathDataSet,  # Path to the new dataset
        epochs=best_params["epochs"],
        batch=best_params["batch"],
        imgsz=best_params["imgsz"],
        optimizer=best_params["optimizer"],
        lr0=best_params["lr0"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"],
        patience=10,  # Early stopping patience
        verbose=True,
        callbacks=[log_metrics]  # Add the callback to log metrics
    )

    print(f"Training log (including metrics and hyperparameters) saved to {metrics_file}")


# Validate the trained model
def validate_model():
    print("Validating YOLOv8 model...")
    subprocess.run([
        'python', '-m', 'ultralytics',
        'task=detect',
        'mode=val',
        f'model=${PathModel}',
        f'data=${PathDataSet}'
    ], check=True)

# Test the model on test data
def test_model():
    print("Testing YOLOv8 model on test data...")
    subprocess.run([
        'python', '-m', 'ultralytics',
        'task=detect',
        'mode=predict',
        f'model=${PathModel}',
        'conf=0.25',
        'source=/scratch/ritali.ai.iitmandi/new',
        '--save'  # Save predictions for inspection
    ], check=True)

PathTestFolder = "/scratch/ritali.ai.iitmandi/new"
PathTestOutPut = "/scratch/ritali.ai.iitmandi/testOutPut"
PathTestOutPutImages = f"{PathTestOutPut}/images"
PathTestOutPutLabels = f"{PathTestOutPut}/labels"

# Test the model on test data
def test_model2():
    print("Testing YOLOv8 model on test data...")

    # Initialize the model
    model = YOLO(PathModel)

    # Perform predictions
    results = model.predict(
        source=PathTestFolder,  # Path to the folder containing test images
        conf=0.25,             # Confidence threshold for predictions
        save=True,             # Save predictions
        save_txt=True,         # Save labels in text format
        project=PathTestOutPutImages,  # Directory to save output images
        name="test_results",         # Sub-directory for current test run
        exist_ok=True                # Allow overwriting existing folder
    )

    # Move the saved label files to the output_labels directory
    import os
    import shutil

    # YOLO saves labels in a subdirectory named 'labels' under the project directory
    predictions_dir = os.path.join(PathTestOutPutImages, "test_results", "labels")
    if os.path.exists(predictions_dir):
        if not os.path.exists(PathTestOutPutLabels):
            os.makedirs(PathTestOutPutLabels)
        for label_file in os.listdir(predictions_dir):
            shutil.move(os.path.join(predictions_dir, label_file), PathTestOutPutLabels)

    print(f"Test results saved in {PathTestOutPutImages}/test_results")
    print(f"Images saved in {PathTestOutPutImages}/test_results")
    print(f"Label files moved to {PathTestOutPutLabels}")


if __name__ == "__main__":
    # Install YOLOv8
    # install_yolov8()

    # # Perform hyperparameter tuning
    best_params = hyperparameter_tuning()

    # # Retrain the model with the best hyperparameters
    build_and_train_model(best_params)

    # # Validate the model
    validate_model()

    # Test the model
    test_model()

    #test with label 
    # test_model2()
