"""
Create composite figure from Grad-CAM visualizations
Combines all 5 Grad-CAM images into a single figure
"""

from PIL import Image
import os

# Setup paths
GRADCAM_DIR = "gradcam_results"
OUTPUT_FILE = "Figure_8_GradCAM_Composite.png"

# Image files
image_files = [
    'gradcam_1_1000_left.jpg',
    'gradcam_2_1000_right.jpg',
    'gradcam_3_1001_left.jpg',
    'gradcam_4_1001_right.jpg',
    'gradcam_5_1002_left.jpg'
]

print("Loading Grad-CAM images...")

# Load all 5 Grad-CAM images
images = []
for img_file in image_files:
    img_path = os.path.join(GRADCAM_DIR, img_file)
    if os.path.exists(img_path):
        images.append(Image.open(img_path))
        print(f"  ✓ Loaded: {img_file}")
    else:
        print(f"  ✗ Not found: {img_file}")

if len(images) == 0:
    print("\nERROR: No images found!")
    print(f"Please ensure images are in: {GRADCAM_DIR}")
    exit(1)

print(f"\nCreating composite from {len(images)} images...")

# Create composite
img_width = images[0].width
img_height = images[0].height
composite = Image.new('RGB', (img_width, img_height * len(images)))

for idx, img in enumerate(images):
    composite.paste(img, (0, idx * img_height))
    print(f"  Added image {idx + 1}/{len(images)}")

# Save high-quality
composite.save(OUTPUT_FILE, quality=95, dpi=(300, 300))
print(f"\n✓ Figure 8 created: {OUTPUT_FILE}")
print(f"  Dimensions: {composite.width}x{composite.height}")
print(f"  File saved successfully!")
