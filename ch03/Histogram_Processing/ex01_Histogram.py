# 구간 선형 변환
# Bit-plane slicing
#  특정 비트의 기여를 강조
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time


def showHist(arrays):
    num = len(arrays)
    # 플롯 설정
    fig, axs = plt.subplots(num, 2, figsize=(12, num * 3))

    for i, (title, array) in enumerate(arrays.items()):
        # 첫 번째 col에 이미지 표시
        axs[i, 0].imshow(array, cmap='gray', vmin=0, vmax=255)
        axs[i, 0].set_title(title)
        axs[i, 0].axis('off')  # 축 표시 없애기

        # 두 번째 col에 히스토그램 표시
        axs[i, 1].hist(array.ravel(), bins=256, color='gray', alpha=1, density=True)
        axs[i, 1].set_xlim([0, 256])

    # 플롯 표시
    plt.tight_layout() 
    script_name = os.path.basename(__file__)
    save_name = os.path.splitext(script_name)[0] + '.png'    
    plt.savefig(f'ch03\\Images\\Result\\Histogram_Processing\\{save_name}', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    image_paths = { 'Dark' : r'ch03\Images\Source\Fig0316(4)(bottom_left).tif',
                    'Bright': r'ch03\Images\Source\Fig0316(1)(top_left).tif',
                    'Low constrastr': r'ch03\Images\Source\Fig0316(2)(2nd_from_top).tif',
                    'Hight-constrast' : r'ch03\Images\Source\Fig0320(3)(third_from_top).tif'}
    arrays = dict()
    for key, path in image_paths.items():
        image = Image.open(path)
        array = np.array(image)
        arrays[key] = array
    
    showHist(arrays)

    