#!/usr/bin/env python
# coding: utf-8

# Installing YOLOv8 module
import subprocess

def install_yolov8():
    print("Installing YOLOv8 module...")
    #subprocess.run(['pip', 'install', 'ultralytics'], check=True)

# Importing necessary libraries
from ultralytics import YOLO

# Model building and training
def build_and_train_model():
    print("Building and training YOLOv8 model...")
    model = YOLO()
    model.train(data="/scratch/ritali.ai.iitmandi/temp/data.yaml", epochs=200,)

def build_train_model_UsingpretrainedWeights():
    print("Retraining YOLOv8 model...")
    subprocess.run([
        'yolo', 
        'task=detect', 
        'mode=train', 
        'model=/scratch/ritali.ai.iitmandi/runs/detect/train7/weights/best.pt', 
        'data=/scratch/ritali.ai.iitmandi/temp/data.yaml',
        'epochs=400',                                            # Specify the number of epochs (adjust as needed)
        'batch=16',                                             # Batch size
        'imgsz=640' 
    ], check=True)

# Validate the trained model
def validate_model():
    print("Validating YOLOv8 model...")
    subprocess.run([
        'yolo', 
        'task=detect', 
        'mode=val', 
        'model=/scratch/ritali.ai.iitmandi/runs/detect/train3/weights/best.pt', 
        'data=/scratch/ritali.ai.iitmandi/temp/data.yaml'
    ], check=True)

# Test the model on test data
def test_model():
    print("Testing YOLOv8 model on test data...")
    subprocess.run([
        'yolo', 
        'task=detect', 
        'mode=predict', 
        'model=/scratch/ritali.ai.iitmandi/runs/detect/train7/weights/best.pt', 
        'conf=0.25', 
        'source=/scratch/ritali.ai.iitmandi/new'
    ], check=True)

if __name__ == "__main__":
    # Install YOLOv8
    # install_yolov8()

    # # Build and train the model
    # build_and_train_model()

    # custom build using pretrained weights

    build_train_model_UsingpretrainedWeights()

    # Validate the model
    # validate_model()

    # Test the model
    # test_model()
