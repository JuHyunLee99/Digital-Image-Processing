import numpy as np
import cv2
import os 

def load_and_transform_image(image_path):
    original_image =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 로그 변환
    c_log = 255 / np.log(1 + np.max(original_image))   # c: 스케일링 상수 =>  표준 8비트 그레이스케일 범위 [0, 255] 벗어나지 않도록 
    log_transformed = c_log * np.log(1 + original_image)
    transformed_image = log_transformed.astype(np.uint8)
    return original_image, transformed_image

def imshow_images(original_image, transformed_image):
    margin = np.full((original_image.shape[0], 10), 255, dtype=np.uint8)
    combined_image = np.hstack((original_image, margin, transformed_image))
    cv2.imshow('Combined Images', combined_image)
    
    while True:
        key = cv2.waitKey()

        # 's' 키가 눌렸다면
        if key == ord('s'):
            # 이미지 저장
            script_name = os.path.basename(__file__)
            save_name = os.path.splitext(script_name)[0] + '_result.jpg'
            cv2.imwrite(f'ch03\\Result\\{save_name}', combined_image)
            print("Image saved as 'saved_image.jpg'")
        else:
            cv2.destroyAllWindows()  # 모든 창을 닫음
            break
   

image_path = r'C:\Source\Digital-Image-Processing\ch03\Images\Fig0305(a)(DFT_no_log).tif'  # 이미지 경로 설정
original_image, log_image = load_and_transform_image(image_path)
imshow_images(original_image, log_image)
