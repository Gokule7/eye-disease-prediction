"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
for Medical Image Interpretability in Eye Disease Classification
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for medical image interpretability
    """
    def __init__(self, model, layer_name=None):
        """
        Args:
            model: trained Keras/TensorFlow model
            layer_name: name of the convolutional layer for visualization
                       If None, will attempt to find the last conv layer
        """
        self.model = model
        
        # Auto-detect last convolutional layer if not specified
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        print(f"Using layer: {layer_name} for Grad-CAM")
        
        # Create submodel that outputs conv layer and final prediction
        try:
            conv_layer = model.get_layer(layer_name)
            # Direct layer access worked
            self.grad_model = tf.keras.models.Model(
                [model.inputs],
                [conv_layer.output, model.output]
            )
        except ValueError:
            # Layer is likely the output of a nested model (like MobileNet)
            # In this case, we use the layer directly as it's already part of the model graph
            for layer in model.layers:
                if layer.name == layer_name:
                    # This layer is a model itself (nested)
                    self.grad_model = tf.keras.models.Model(
                        [model.inputs],
                        [layer.output, model.output]
                    )
                    return
            raise ValueError(f"Could not create GradCAM model for layer: {layer_name}")
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        # First try direct layers
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        
        # If no conv layer found, check if first layer is a Model (like pre-trained base)
        if len(self.model.layers) > 0:
            first_layer = self.model.layers[0]
            if hasattr(first_layer, 'layers'):
                # This is a nested model (e.g., pre-trained base)
                for layer in reversed(first_layer.layers):
                    if 'conv' in layer.name.lower():
                        # Return the full path
                        return layer.name
        
        raise ValueError("No convolutional layer found in model")
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap for input image
        
        Args:
            img_array: preprocessed image (1, H, W, 3) normalized to [0,1]
            pred_index: class index (if None, uses model's predicted class)
        
        Returns:
            heatmap: normalized attention map
            pred_index: predicted class index
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            
            # If pred_index not specified, use predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients with respect to conv layer output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute weight: average gradient values across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the activation map by the importance of each channel
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap to 0-1 range
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy(), int(pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index)
    
    def overlay_heatmap(self, original_img, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            original_img: original fundus image (H, W, 3) in [0,1]
            heatmap: Grad-CAM heatmap
            alpha: transparency of overlay
        
        Returns:
            overlaid_img: heatmap overlaid on original
            heatmap_resized: resized heatmap
        """
        img_height, img_width = original_img.shape[:2]
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
        
        # Convert to colormap (jet for medical visualization)
        cmap = get_cmap('jet')
        heatmap_colored = cmap(heatmap_resized)[..., :3]  # RGB only
        
        # Ensure original image is in correct format
        if original_img.max() > 1:
            original_img = original_img / 255.0
        
        # Blend images
        overlaid = (1 - alpha) * original_img + alpha * heatmap_colored
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid, heatmap_resized
    
    def visualize(self, img_array, original_img, class_names, pred_index=None, save_path=None):
        """
        Create visualization with original + heatmap + overlay
        
        Args:
            img_array: preprocessed image (1, H, W, 3)
            original_img: original fundus image (H, W, 3)
            class_names: list of disease class names
            pred_index: class to visualize
            save_path: path to save the figure
        
        Returns:
            fig: matplotlib figure object
            predictions: model predictions
            pred_class: predicted class index
        """
        heatmap, pred_class = self.generate_heatmap(img_array, pred_index)
        overlaid, heatmap_resized = self.overlay_heatmap(original_img, heatmap)
        
        # Get prediction probabilities
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Fundus Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Attention Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(overlaid)
        disease = class_names[pred_class] if pred_class < len(class_names) else "Unknown"
        confidence = predictions[pred_class] * 100
        axes[2].set_title(f'Prediction: {disease}\nConfidence: {confidence:.1f}%', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig, predictions, pred_class


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model input
    
    Args:
        img_path: path to image file
        target_size: tuple (height, width) for resizing
    
    Returns:
        img_array: preprocessed image array (1, H, W, 3)
        original_img: original image for visualization (H, W, 3)
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_normalized, 0)
    
    return img_array, img_normalized


def generate_gradcam_for_images(model, image_paths, class_names, output_dir="gradcam_results", 
                                layer_name=None, target_size=(224, 224)):
    """
    Generate Grad-CAM visualizations for multiple images
    
    Args:
        model: trained model
        image_paths: list of tuples (image_path, class_name, class_index)
        class_names: list of all class names
        output_dir: directory to save results
        layer_name: specific layer to use for Grad-CAM
        target_size: input size for the model
    
    Returns:
        results: list of dictionaries with results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, layer_name=layer_name)
    
    results = []
    
    for idx, (img_path, true_class, true_idx) in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {img_path}")
        
        try:
            # Load and preprocess image
            img_array, img_normalized = load_and_preprocess_image(img_path, target_size)
            
            # Generate Grad-CAM
            img_name = os.path.basename(img_path)
            save_path = os.path.join(output_dir, f"gradcam_{true_class}_{idx+1}.png")
            
            fig, predictions, pred_class = gradcam.visualize(
                img_array,
                img_normalized,
                class_names,
                pred_index=None,  # Use model's prediction
                save_path=save_path
            )
            
            # Store results
            result = {
                'image': img_name,
                'true_class': true_class,
                'true_index': true_idx,
                'predicted_class': class_names[pred_class],
                'predicted_index': pred_class,
                'confidence': predictions[pred_class] * 100,
                'correct': (pred_class == true_idx),
                'save_path': save_path
            }
            results.append(result)
            
            # Print results
            print(f"  True: {true_class} | Predicted: {class_names[pred_class]} | " +
                  f"Confidence: {predictions[pred_class]*100:.1f}%")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error processing {img_path}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Grad-CAM Generation Complete!")
    print(f"Total images processed: {len(results)}")
    print(f"Correct predictions: {sum(r['correct'] for r in results)}/{len(results)}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print("Grad-CAM Visualization Module")
    print("This module provides Grad-CAM functionality for model interpretability")
    print("Use run_gradcam.py to generate visualizations")
