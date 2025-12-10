"""
Simple Grad-CAM Demonstration for Eye Disease Images
This creates visualizations showing where the model focuses its attention
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def create_simple_model(input_shape=(224, 224, 3)):
    """Create a simple CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    grads = tape.gradient(class_channel, conv_outputs)
    
    # This is a vector where each entry is the mean intensity of the gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), int(pred_index.numpy())


def create_visualization(original_img, heatmap, predictions, class_names, pred_class):
    """Create the 3-panel visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Fundus Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    im = axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Attention Map', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    cmap = get_cmap('jet')
    heatmap_colored = cmap(heatmap_resized)[..., :3]
    overlaid = 0.6 * original_img + 0.4 * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    
    axes[2].imshow(overlaid)
    confidence = predictions[pred_class] * 100
    axes[2].set_title(f'Prediction: {class_names[pred_class]}\nConfidence: {confidence:.1f}%',
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    print("="*70)
    print("Grad-CAM Visualization Demo for Eye Disease Classification")
    print("="*70)
    
    # Setup paths
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    TESTING_DIR = os.path.join(PROJECT_DIR, "ODIR-5K", "ODIR-5K", "Testing Images")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "gradcam_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    INPUT_SIZE = (224, 224)
    class_names = ['Normal', 'Disease']
    
    # Find test images
    print(f"\nSearching for test images in: {TESTING_DIR}")
    if not os.path.exists(TESTING_DIR):
        print(f"ERROR: Directory not found: {TESTING_DIR}")
        return
    
    image_files = [f for f in os.listdir(TESTING_DIR) if f.endswith(('.jpg', '.png'))][:5]
    print(f"Found {len(image_files)} test images")
    
    # Create model
    print("\nCreating demo CNN model...")
    model = create_simple_model(input_shape=(*INPUT_SIZE, 3))
    print("✓ Model created successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Use the last convolutional layer
    last_conv_layer_name = 'conv4'
    print(f"  Using layer '{last_conv_layer_name}' for Grad-CAM")
    
    # Process each image
    print(f"\nGenerating Grad-CAM visualizations...")
    print("="*70)
    
    successful = 0
    for idx, img_file in enumerate(image_files, 1):
        try:
            img_path = os.path.join(TESTING_DIR, img_file)
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_file}")
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, INPUT_SIZE)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_normalized, 0)
            
            # Get prediction
            predictions = model.predict(img_array, verbose=0)[0]
            pred_class = np.argmax(predictions)
            confidence = predictions[pred_class] * 100
            
            print(f"  Prediction: {class_names[pred_class]} ({confidence:.1f}%)")
            
            # Generate Grad-CAM heatmap
            heatmap, pred_idx = generate_gradcam_heatmap(
                model, img_array, last_conv_layer_name
            )
            
            # Create visualization
            fig = create_visualization(
                img_normalized, heatmap, predictions, class_names, pred_class
            )
            
            # Save
            save_path = os.path.join(OUTPUT_DIR, f"gradcam_{idx}_{img_file}")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved: {save_path}")
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print(f"Grad-CAM Demonstration Complete!")
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
    
    print("\n" + "="*70)
    print("NOTE:")
    print("- This demo uses a simple CNN model for demonstration")
    print("- The model is untrained, so predictions are random")
    print("- Grad-CAM still shows which image regions influence predictions")
    print("-" * 70)
    print("For actual eye disease classification:")
    print("  1. Train your model using the provided notebook")
    print("  2. Save the trained model (.h5 or .keras file)")
    print("  3. Update run_gradcam.py to load your trained model")
    print("  4. Run run_gradcam.py for real predictions with Grad-CAM")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
