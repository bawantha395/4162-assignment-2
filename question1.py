import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('images', exist_ok=True)

# Image size
img_size = 150
img = np.zeros((img_size, img_size), dtype=np.uint8)  # Background = 0

# Draw Object 1: Square (gray value 128) 
cv2.rectangle(img, (20, 20), (60, 60), 128, -1)

# Draw Object 2: Circle (gray value 255) 
cv2.circle(img, (110, 110), 25, 255, -1)

# Add Gaussian noise
mean = 0
stddev = 20
noise = np.random.normal(mean, stddev, img.shape).astype(np.int16)
noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


smoothed = cv2.GaussianBlur(noisy_img, (5, 5), 0)

# Apply Otsu's threshold
_, otsu_thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save images
cv2.imwrite('images/original.png', img)
cv2.imwrite('images/noisy.png', noisy_img)
cv2.imwrite('images/otsu.png', otsu_thresh)

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original (Square First, Circle Second)")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Otsu Threshold Result")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.savefig('images/otsu_results.png')
plt.show()
