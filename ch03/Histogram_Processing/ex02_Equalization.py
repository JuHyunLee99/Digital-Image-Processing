# 히스토그램 평활화
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def showHist(arrays):
    num = len(arrays)
    # 플롯 설정
    fig, axs = plt.subplots(num, 4, figsize=(18, 8))
    fig.subplots_adjust(left=0, right=0.93, top=0.95, bottom=0.05, hspace=0.2, wspace=0)
    for i, (title, array) in enumerate(arrays.items()):
        # 첫 번째 col에 이미지 표시
        axs[i, 0].imshow(array[0], cmap='gray', vmin=0, vmax=255)
        axs[i, 0].set_title(title)
        axs[i, 0].axis('off')  # 축 표시 없애기
        
        # 두 번째 col에 원본 히스토그램 표시
        axs[i, 1].hist(array[0].ravel(), bins=256, color='black',density=True)
        axs[i, 1].set_xlim([-1, 256])

        # 두 번째 col에 히스토그램 평활화 이미지 표시
        axs[i, 2].imshow(array[1], cmap='gray', vmin=0, vmax=255)
        axs[i, 2].set_title(title)
        axs[i, 2].axis('off')  
        
        # 세 번째 col에 평활화 히스토그램 표시
        axs[i, 3].hist(array[1].ravel(), bins=256, color='black', density=True)
        axs[i, 3].set_xlim([-1, 256])

    # 플롯 표시
    # plt.tight_layout() 
    script_name = os.path.basename(__file__)
    save_name = os.path.splitext(script_name)[0] + '.png'    
    plt.savefig(f'ch03\\Images\\Result\\Histogram_Processing\\{save_name}', bbox_inches='tight')
    plt.show()

def histogram_equalizatioin(origin_array):
    # 히스토그램 계산
    hist, bins = np.histogram(origin_array, 256, [0,256])
    hist_normalized = hist / hist.sum()
    
    # 누적 분포 함수(CDF) 계산
    cdf = hist_normalized.cumsum()
    
    cdf_scaled = np.round(cdf * 255).astype(np.uint8)
    
    # ※ 다차원 배열을 1차원으로 만들기
    # flatten은 복사본을 생성 
    # ravel은 참조
    
    # 원본 배열의 픽셀 값에 CDF를 매핑
    equalized_array = cdf_scaled[origin_array.flatten()].reshape(origin_array.shape)    
    return equalized_array
    
if __name__ == "__main__":
    image_paths = { 'Dark' : r'ch03\Images\Source\Fig0316(4)(bottom_left).tif',
                    'Bright': r'ch03\Images\Source\Fig0316(1)(top_left).tif',
                    'Low constrastr': r'ch03\Images\Source\Fig0316(2)(2nd_from_top).tif',
                    'Hight-constrast' : r'ch03\Images\Source\Fig0320(3)(third_from_top).tif'}
    arrays = dict()
    for key, path in image_paths.items():
        origin_image = Image.open(path)
        origin_array = np.array(origin_image)
        equalizated_array = histogram_equalizatioin(origin_array)
        arrays[key] = (origin_array, equalizated_array)
       
    showHist(arrays)