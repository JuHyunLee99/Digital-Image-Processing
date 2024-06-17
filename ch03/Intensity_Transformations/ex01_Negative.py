from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
   
image_path = r'C:\Source\Digital-Image-Processing\ch03\Images\Fig0304(a)(breast_digital_Xray).tif'  # 이미지 경로 설정
original_image = Image.open(image_path)
# original_image.show()
# print("Image mode:", original_image.mode)
original_array = np.array(original_image)

 # 네거티브 변환 수행
max_value = np.iinfo(original_array.dtype).max
min_value = np.iinfo(original_array.dtype).min
negative_array = max_value - original_array
negative_image = Image.fromarray(negative_array, 'L')
# negative_image.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray', vmin=min_value, vmax=max_value)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_array, cmap='gray', vmin=min_value, vmax=max_value)
plt.title('Negative_image')
plt.axis('off')   
# plt.show()

spacing = 10  # 픽셀 단위
total_width = original_image.width * 2 + spacing
max_height = original_image.height
new_image = Image.new('L', (total_width, max_height))
# 새 캔버스에 이미지 복사
new_image.paste(original_image, (0, 0))  # 원본 이미지를 왼쪽에 붙임
new_image.paste(negative_image, (original_image.width+spacing, 0))  # 처리된 이미지를 오른쪽에 붙임

# 이미지 저장 또는 표시
new_image.show()  # 이미지 보기
new_image.save('ch03\Result\ex01_Negative.tif')  # 이미지 저장