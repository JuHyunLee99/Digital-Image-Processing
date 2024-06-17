# 구간 선형 변환
# 콘트라스트 스트레칭
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_and_transform_image(image_path):
    original_image = Image.open(image_path)
    original_image.show()
    image_array = np.array(original_image, dtype=np.float32)
    transformed_images = []

    min_val = np.min(image_array)
    max_val = np.max(image_array)

    print(image_array.dtype)
    print(image_array.min(), image_array.max())
    # 선형 스트레칭 수행
    stretched_image = (image_array - min_val) / (max_val - min_val) * 255
    transformed_images.append(stretched_image.astype(np.uint8))  # 데이터 타입을 uint8로 변경

    return original_image, transformed_images

def plot_images(original, transformed_images):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray', vmin=91.0, vmax=138.0)
    plt.title('Original(Low-Contrast)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(transformed_images[0], cmap='gray')
    plt.title('Contrast Stretching')
    plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(transformed_images[1], cmap='gray')
    # plt.title('Thresholding')
    # plt.axis('off')

    plt.show()

image_path = r'C:\Source\Digital-Image-Processing\ch03\Images\Fig0310(b)(washed_out_pollen_image).tif'  # 이미지 경로 설정
# original_image, transformed_images = load_and_transform_image(image_path)
# cv2.namedWindow('image')
# image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# cv2.imshow("image",image)
# cv2.waitKey()
# cv2.destroyAllWindows()
# original_image = Image.open(image_path)
# image_array = np.array(original_image)
# plt.figure()
# plt.imshow(image_array, cmap='gray')
# plt.show()
# plot_images(original_image, transformed_images)

image_pil = Image.open(image_path)
image_pil.show()
image = np.array(image_pil)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()