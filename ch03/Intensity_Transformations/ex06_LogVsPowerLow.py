# 로그 변환, 감마 변환 차이

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_transform_image(image_path):
    original_image = Image.open(image_path)
    image_array = np.array(original_image, dtype=float)

    # 로그 변환
    c_log = 255 / np.log(1 + np.max(image_array))
    log_transformed = c_log * np.log(1 + image_array)

    # 감마 변환
    gamma = 0.4
    c_gamma = 255 / np.power(np.max(image_array), gamma)
    gamma_transformed = c_gamma * np.power(image_array, gamma)

    return original_image, log_transformed, gamma_transformed

def plot_images(original, log_transformed, gamma_transformed):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(log_transformed, cmap='gray')
    plt.title('Log Transformed Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gamma_transformed, cmap='gray')
    plt.title('Gamma Transformed Image (γ=0.4)')
    plt.axis('off')

    plt.show()

image_path = r'C:\Users\ACTRO\Downloads\DIP3E_CH03_Original_Images\DIP3E_Original_Images_CH03\Fig0308(a)(fractured_spine).tif'  # 이미지 경로 설정
original_image, log_transformed, gamma_transformed = load_and_transform_image(image_path)
plot_images(original_image, log_transformed, gamma_transformed)
