# 콘트라스트 개선
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_transform_image(image_path):
    original_image = Image.open(image_path)
    image_array = np.array(original_image, dtype=float)
    transformed_images = []
    
    gamma = 0.6
    c_gamma = 255 / np.power(np.max(image_array), gamma)
    gamma_transformed = c_gamma * np.power(image_array, gamma)
    transformed_images.append(Image.fromarray(gamma_transformed))

    gamma = 0.4
    c_gamma = 255 / np.power(np.max(image_array), gamma)
    gamma_transformed = c_gamma * np.power(image_array, gamma)
    transformed_images.append(Image.fromarray(gamma_transformed))


    gamma = 0.3
    c_gamma = 255 / np.power(np.max(image_array), gamma)
    gamma_transformed = c_gamma * np.power(image_array, gamma)
    transformed_images.append(Image.fromarray(gamma_transformed))

    return original_image, transformed_images

def plot_images(original_image, transformed_images):
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(transformed_images[0], cmap='gray')
    plt.title('γ = 0.6')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(transformed_images[1], cmap='gray')
    plt.title('γ = 0.4')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(transformed_images[2], cmap='gray')
    plt.title('γ = 0.3')
    plt.axis('off')
    plt.show()

image_path = r'ch03\Images\Fig0308(a)(fractured_spine).tif'  # 이미지 경로 설정
original_image, transformed_images = load_and_transform_image(image_path)
plot_images(original_image, transformed_images)
