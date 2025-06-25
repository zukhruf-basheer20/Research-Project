import cv2
import os
import numpy as np

INPUT_DIR = '../data/Leaf_Final_Segmented'  # adjust path to your local dir
OUTPUT_DIR = '../data/Leaf_Clean_Sketch'  # output dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    filepath = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to smooth out noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to isolate leaf
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if needed (leaf might be black on white or white on black)
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in {filename}")
        continue

    # Get largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create black canvas
    outline = np.zeros_like(gray)

    # Draw white contour on black
    cv2.drawContours(outline, [largest_contour], -1, 255, thickness=2)

    # Save final outlined image
    out_path = os.path.join(OUTPUT_DIR, filename.replace('.png', '_outline.png'))
    cv2.imwrite(out_path, outline)

    print(f"Processed: {filename} → {out_path}")
