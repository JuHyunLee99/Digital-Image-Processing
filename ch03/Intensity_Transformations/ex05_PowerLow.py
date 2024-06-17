# 콘트라스트 개선 (γ > 1)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --------------------------Power Law Contrast Enhancement (γ > 1)--------------------
def gammaTransform(gamma, original_array):
    c_gamma = 255 / np.power(np.max(original_array), gamma)
    gamma_array = c_gamma * np.power(original_array, gamma)
    gamma_array = gamma_array.astype(np.uint8)
    return gamma_array
# -------------------------------------------------------------------------------

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
    
if __name__ == "__main__":
    image_path = r'ch03\Images\Source\Fig0309(a)(washed_out_aerial_image).tif' 
    original_image = Image.open(image_path)
    original_array = np.array(original_image, dtype=np.float32)   

    images = { "Original Image" : original_image}
    gammas = [3, 4, 5]
    for gamma in gammas:
        log_array = gammaTransform(gamma, original_array)
        log_Image = Image.fromarray(log_array, original_image.mode)
        images[f'γ = {gamma}'] = log_Image
        
    showImages(images, 1, 4, title= 'Power Law Contrast Enhancement (γ > 1)')
