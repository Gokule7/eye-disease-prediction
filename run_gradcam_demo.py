"""
Demo Grad-CAM Visualization with Pre-trained Model
This script demonstrates Grad-CAM functionality using a pre-trained model
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gradcam_visualization import GradCAM, load_and_preprocess_image


def create_demo_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a simple CNN model for demonstration purposes
    In practice, you should load your trained model
    """
    print("Creating demo model (for demonstration purposes)...")
    
    # Use a pre-trained model for better results
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Build the full model using Functional API for better compatibility
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Demo model created")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    return model, base_model


def find_test_images(testing_dir, num_images=5):
    """Find available test images in the directory"""
    if not os.path.exists(testing_dir):
        return []
    
    image_files = [f for f in os.listdir(testing_dir) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Select first num_images
    selected = image_files[:num_images]
    return [os.path.join(testing_dir, f) for f in selected]


def generate_demo_gradcam_visualizations():
    """Generate Grad-CAM visualizations for demo images"""
    
    print("="*70)
    print("Grad-CAM Demonstration for Eye Disease Classification")
    print("="*70)
    
    # Configuration
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TESTING_DIR = os.path.join(PROJECT_DIR, "ODIR-5K", "ODIR-5K", "Testing Images")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "gradcam_results")
    INPUT_SIZE = (224, 224)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find test images
    print(f"\nSearching for test images in: {TESTING_DIR}")
    test_image_paths = find_test_images(TESTING_DIR, num_images=5)
    
    if len(test_image_paths) == 0:
        print("ERROR: No test images found!")
        print("Please ensure test images exist in:", TESTING_DIR)
        return
    
    print(f"Found {len(test_image_paths)} test images:")
    for path in test_image_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Create demo model
    model, base_model = create_demo_model(input_shape=(*INPUT_SIZE, 3), num_classes=2)
    
    # Class names for binary classification
    class_names = ['Normal', 'Disease']
    
    # Use the output of the base model (last layer before pooling)
    # The mobilenetv2 layer contains all the conv layers
    base_model_layer_name = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model_layer_name = layer.name
            break
    
    if base_model_layer_name is None:
        print("ERROR: Could not find base model layer")
        return
    
    print(f"\nUsing base model layer: {base_model_layer_name}")
    
    # Initialize Grad-CAM - it will work with the output of this layer
    try:
        gradcam = GradCAM(model, layer_name=base_model_layer_name)
        print(f"✓ Grad-CAM initialized")
    except Exception as e:
        print(f"ERROR: Could not initialize Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nGenerating Grad-CAM visualizations...")
    print("="*70)
    
    # Process each image
    successful = 0
    for idx, img_path in enumerate(test_image_paths, 1):
        try:
            img_name = os.path.basename(img_path)
            print(f"\n[{idx}/{len(test_image_paths)}] Processing: {img_name}")
            
            # Load and preprocess image
            img_array, img_normalized = load_and_preprocess_image(img_path, INPUT_SIZE)
            
            # Generate prediction
            predictions = model.predict(img_array, verbose=0)[0]
            pred_class = np.argmax(predictions)
            confidence = predictions[pred_class] * 100
            
            print(f"  Prediction: {class_names[pred_class]} ({confidence:.1f}%)")
            
            # Generate Grad-CAM visualization using our GradCAM class
            fig, _, _ = gradcam.visualize(
                img_array,
                img_normalized,
                class_names,
                pred_index=None
            )
            
            # Save figure
            save_path = os.path.join(OUTPUT_DIR, f"gradcam_demo_{idx}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved: {save_path}")
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue

    
    # Print summary
    print("\n" + "="*70)
    print(f"Grad-CAM Demonstration Complete!")
    print(f"Successfully processed: {successful}/{len(test_image_paths)} images")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
    
    print("\n" + "="*70)
    print("NOTE: This demonstration uses a pre-trained MobileNetV2 model")
    print("For actual eye disease classification, please:")
    print("  1. Train your model using the notebook")
    print("  2. Save the trained model (.h5 or .keras)")
    print("  3. Update run_gradcam.py to use your trained model")
    print("="*70)


if __name__ == "__main__":
    try:
        generate_demo_gradcam_visualizations()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
