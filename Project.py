import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

image_url = 'https://media.vanityfair.com/photos/65aa9c2280771f83d1a6f1fe/master/w_1920,c_limit/Taylor-Swift.jpg'
image = io.imread(image_url)
gray_image = color.rgb2gray(image)

U, Sigma, VT = np.linalg.svd(gray_image, full_matrices=False)

def reconstruct_image(U, Sigma, VT, k):
    return np.dot(U[:, :k], np.dot(np.diag(Sigma[:k]), VT[:k, :]))

k_values = [5, 20, 50, 100]

fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(15, 5))

axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

for i, k in enumerate(k_values):
    compressed_image = reconstruct_image(U, Sigma, VT, k)
    axes[i + 1].imshow(compressed_image, cmap='gray')
    axes[i + 1].set_title(f'k = {k}')
    axes[i + 1].axis('off')

plt.tight_layout()
plt.show()

def compression_ratio(original_shape, k):
    m, n = original_shape
    return (k * (m + n + 1)) / (m * n)

def reconstruction_error(original, reconstructed):
    return np.linalg.norm(original - reconstructed)

for k in k_values:
    compressed_image = reconstruct_image(U, Sigma, VT, k)
    print(f'k = {k}, Compression Ratio = {compression_ratio(gray_image.shape, k):.2f}, Reconstruction Error = {reconstruction_error(gray_image, compressed_image):.2f}')
