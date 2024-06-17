# 감마 보정
import numpy as np
import cv2
import os

def load_and_transform_image(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 감마 변환
    gamma = 0.4
    c_gamma = 255 / np.power(np.max(original_image), gamma)
    gamma_transformed = c_gamma * np.power(original_image, gamma)
    transformed_image = gamma_transformed.astype(np.uint8)
    return original_image, transformed_image

def imshow_images(original_image, transformed_image):
    margin = np.full((50, original_image.shape[1]), 255, dtype=np.uint8)
    original_image =  np.vstack((margin, original_image))
    transformed_image = np.vstack((margin, transformed_image))
    cv2.putText(original_image, "original_image", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , 0, 2, cv2.LINE_AA)
    cv2.putText(transformed_image, "Gamma_image (0.4)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , 0, 2, cv2.LINE_AA)
    
    margin = np.full((original_image.shape[0], 50), 255, dtype=np.uint8)
    combined_image = np.hstack((margin, original_image, margin, transformed_image, margin))
    margin = np.full((50, combined_image.shape[1]), 255, dtype=np.uint8)
    combined_image = np.vstack((margin,margin, combined_image, margin))
    
    
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
        
image_path = r'ch03\Images\Fig0307(a)(intensity_ramp).tif'  # 이미지 경로 설정
original_image, transformed_image = load_and_transform_image(image_path)
imshow_images(original_image, transformed_image)
