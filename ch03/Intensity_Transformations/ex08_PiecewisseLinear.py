# 구간 선형 변환
# 밝기-레벨 슬라이싱
# 특정 밝기 범위 강조
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 텍스트 바운딩 박스 계산
def getTextSize(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return (text_width, text_height)

# 결과 이미지 나타내기  
def showImages(images, row, col, spacing = 20, fontSize = 30, title = None, indexNewImg = -1):
    # 새 캔버스 만들기
    original_image = list(images.values())[0]
    total_width = (original_image.width + spacing) * col + spacing
    if title == None:
        total_height = (original_image.height + 2*spacing) * row + 2*spacing
    else:
        total_height = (original_image.height + 2*spacing) * row + 4*spacing
    new_image = Image.new('L', (total_width, total_height), "White")

    # 새 캔버스에 이미지 복사, 텍스트 넣기
    draw = ImageDraw.Draw(new_image)
    # Title text
    if title != None:
        font = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', fontSize*1.3)    
        text_width = getTextSize(draw, title, font)[0]
        draw.text(((new_image.width - text_width) // 2, spacing//2), title, font=font, fill="black")
    
    font = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', fontSize)    
    for index, (text, image) in enumerate(images.items()):
        #  이미지 위치  
        if (index%col) != 0 :
            img_x += original_image.width + spacing
        else:
            img_x = spacing
            if index == 0:
                if title == None:
                    img_y = 3*spacing
                else: 
                    img_y = 5*spacing
            else:
                img_y +=  original_image.height + 2*spacing
                
        # 이미지를 왼쪽에 붙임
        new_image.paste(image, (img_x, img_y))  
        
        # 텍스트 위치
        text_width = getTextSize(draw, text, font)[0]
        text_x = (original_image.width - text_width) // 2 + img_x
        text_y = img_y - 2*spacing
        draw.text((text_x, text_y), text, font=font, fill="black")
        
    new_image.show()
    script_name = os.path.basename(__file__)
    if indexNewImg == -1:
        save_name = os.path.splitext(script_name)[0] + '.png'
    else:
        save_name = os.path.splitext(script_name)[0] + f'_{indexNewImg+1}' + '.png'        
    new_image.save(f'ch03\\Images\\Result\\{save_name}')
    time.sleep(0.5)   
    

# ----------------- Piecewise-Linear Transformation ------------------------
def intensityLevelSlicing(original_array, lower, upper, binary_mode):
    
    if binary_mode:
        sliced_array = np.where((original_array >= lower) & (original_array <= upper), 255, 0)
    else:
        sliced_array = np.where((original_array >= lower) & (original_array <= upper), 0, original_array)
        
    sliced_array = sliced_array.astype(np.uint8)
    return sliced_array
# -------------------------------------------------------------------------

if __name__ == "__main__":
    image_path = r'ch03\Images\Source\Fig0312(a)(kidney).tif'
    original_image = Image.open(image_path)
    original_array = np.array(original_image, dtype=np.float32)   
    images = { "Original Image" : original_image}
    
    # Intensity Level Slicing
    # 구간을 어디로 할지는 나중에 더 배우고 제대로....
    
    # 관심영역 255, 나머지 0
    sliced_array = intensityLevelSlicing(original_array, 150, 255, True)
    sliced_Image = Image.fromarray(sliced_array, original_image.mode)
    images['s = 255 if 150 ≤ r ≤ 255, otherwise s = 0'] = sliced_Image
    
    # 관심영역 0, 나머지 그대로 유지
    sliced_array = intensityLevelSlicing(original_array, 0, 150, False)
    sliced_Image = Image.fromarray(sliced_array, original_image.mode)
    images['s = 0 if 0 ≤ r ≤ 150, otherwise s = r'] = sliced_Image
       
    showImages(images, 1, 3, title= 'Piecewise-Linear Transformation (Intensity-Level-Slicing)')

    