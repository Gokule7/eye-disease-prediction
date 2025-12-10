"""
Run Grad-CAM on Eye Disease Test Images
Generates visualization heatmaps showing model attention regions
"""

import os
import sys
import pandas as pd
import tensorflow as tf
from gradcam_visualization import generate_gradcam_for_images, load_and_preprocess_image, GradCAM
import matplotlib.pyplot as plt
import numpy as np


def find_model_file():
    """Search for trained model file in the project directory"""
    possible_extensions = ['.h5', '.keras', '.hdf5']
    possible_names = ['best_model', 'model', 'eye_disease', 'efficientnet']
    
    # Search in current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.endswith(ext) for ext in possible_extensions):
                return os.path.join(root, file)
    
    return None


def select_test_images_from_csv(csv_path, testing_dir, num_per_class=2):
    """
    Select representative test images for each disease class
    
    Args:
        csv_path: path to full_df.csv
        testing_dir: directory containing test images
        num_per_class: number of images per class
    
    Returns:
        list of tuples (image_path, class_name, class_index)
    """
    df = pd.read_csv(csv_path)
    
    # Disease categories based on the dataset
    # N=Normal, D=Diabetic Retinopathy, G=Glaucoma, C=Cataract, A=ARMD
    disease_columns = ['N', 'D', 'G', 'C', 'A']
    disease_names = {
        'N': 'Normal',
        'D': 'Diabetic_Retinopathy',
        'G': 'Glaucoma',
        'C': 'Cataract',
        'A': 'ARMD'
    }
    
    selected_images = []
    
    for idx, disease_col in enumerate(disease_columns):
        # Filter images where this disease is positive
        disease_df = df[df[disease_col] == 1]
        
        if len(disease_df) == 0:
            print(f"No images found for {disease_names[disease_col]}")
            continue
        
        # Select up to num_per_class images
        sample_df = disease_df.head(num_per_class)
        
        for _, row in sample_df.iterrows():
            filename = row['filename']
            img_path = os.path.join(testing_dir, filename)
            
            # Check if file exists in Testing Images
            if not os.path.exists(img_path):
                # Try Training Images as fallback
                img_path = img_path.replace('Testing Images', 'Training Images')
            
            if os.path.exists(img_path):
                selected_images.append((
                    img_path,
                    disease_names[disease_col],
                    idx
                ))
                print(f"Selected: {disease_names[disease_col]} - {filename}")
            else:
                print(f"Warning: Image not found: {filename}")
    
    return selected_images


def manually_select_test_images(testing_dir):
    """
    Manually select diverse test images from the Testing Images folder
    Returns list of tuples (image_path, class_name, class_index)
    """
    # Select specific images that represent different conditions
    # These are examples - adjust based on what's available in your dataset
    
    # For binary classification (X vs N), we'll use class 0 and 1
    # But label them meaningfully
    selected_images = []
    
    test_images_list = [
        # Normal eyes
        ('1001_left.jpg', 'Normal', 0),
        ('1007_right.jpg', 'Normal', 0),
        # Disease cases (treated as class 1)
        ('1000_left.jpg', 'Disease', 1),
        ('1002_right.jpg', 'Disease', 1),
        ('1003_left.jpg', 'Disease', 1),
    ]
    
    for filename, label, class_idx in test_images_list:
        img_path = os.path.join(testing_dir, filename)
        if os.path.exists(img_path):
            selected_images.append((img_path, label, class_idx))
            print(f"Selected: {label} - {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    return selected_images


def main():
    """Main function to run Grad-CAM visualization"""
    
    print("="*60)
    print("Grad-CAM Visualization for Eye Disease Classification")
    print("="*60)
    
    # Configuration
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TESTING_DIR = os.path.join(PROJECT_DIR, "ODIR-5K", "ODIR-5K", "Testing Images")
    CSV_PATH = os.path.join(PROJECT_DIR, "full_df.csv")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "gradcam_results")
    MODEL_PATH = os.path.join(PROJECT_DIR, "brain_tumors", "best_model.keras")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found at: {MODEL_PATH}")
        print("Searching for model file...")
        MODEL_PATH = find_model_file()
        
        if MODEL_PATH is None:
            print("\nERROR: No trained model found!")
            print("Please ensure you have a trained model file (.h5, .keras, or .hdf5)")
            print("You can train a model using the notebook first.")
            return
    
    print(f"\nUsing model: {MODEL_PATH}")
    
    # Load the trained model
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Determine input size
        input_shape = model.input_shape
        target_size = (input_shape[1], input_shape[2])
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return
    
    # Determine class names based on model output
    num_classes = model.output_shape[-1]
    if num_classes == 2:
        # Binary classification
        class_names = ['Normal (X)', 'Disease (N)']
        print(f"\nDetected binary classification model")
    elif num_classes == 8:
        # Multi-class (original ODIR classes)
        class_names = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 
                       'ARMD', 'Hypertensive', 'Myopia', 'Others']
        print(f"\nDetected multi-class classification model ({num_classes} classes)")
    else:
        class_names = [f'Class_{i}' for i in range(num_classes)]
        print(f"\nDetected {num_classes}-class model")
    
    # Select test images
    print(f"\nSelecting test images from: {TESTING_DIR}")
    
    if os.path.exists(CSV_PATH):
        print("Using CSV to select representative images...")
        test_images = select_test_images_from_csv(CSV_PATH, TESTING_DIR, num_per_class=1)
    else:
        print("CSV not found. Using manual selection...")
        test_images = manually_select_test_images(TESTING_DIR)
    
    if len(test_images) == 0:
        print("\nERROR: No test images found!")
        print("Please check that test images exist in:", TESTING_DIR)
        return
    
    print(f"\nTotal images selected: {len(test_images)}")
    
    # Generate Grad-CAM visualizations
    print(f"\nGenerating Grad-CAM visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    try:
        results = generate_gradcam_for_images(
            model=model,
            image_paths=test_images,
            class_names=class_names,
            output_dir=OUTPUT_DIR,
            layer_name=None,  # Auto-detect last conv layer
            target_size=target_size
        )
        
        # Print detailed results
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['image']}")
            print(f"   True Class: {result['true_class']}")
            print(f"   Predicted: {result['predicted_class']} ({result['confidence']:.1f}%)")
            print(f"   Correct: {'✓' if result['correct'] else '✗'}")
            print(f"   Saved: {result['save_path']}")
        
        accuracy = sum(r['correct'] for r in results) / len(results) * 100
        print(f"\n{'='*60}")
        print(f"Accuracy on test set: {accuracy:.1f}%")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR during Grad-CAM generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✓ Grad-CAM visualization complete!")
    print(f"  Check the '{OUTPUT_DIR}' folder for results")


if __name__ == "__main__":
    main()
