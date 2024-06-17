from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def getTextSize(draw, text, font):
    # 텍스트 바운딩 박스 계산
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return (text_width, text_height)  

image_path = r'ch03\Images\Source\Fig0305(a)(DFT_no_log).tif'  # 이미지 경로 설정
original_image = Image.open(image_path)
original_array = np.array(original_image, dtype=np.float32)

# -------------------------- 로그 변환 수행 --------------------------
c_log = 255 / np.log(1 + np.max(original_array))   # c: 스케일링 상수 =>  표준 8비트 그레이스케일 범위 [0, 255] 벗어나지 않도록 
log_array = c_log * np.log(1 + original_array) # np.log(1 + original_array)에서 256으로 오버플로우 발생하므로 np.array(dtype=np.float32)으로 설정.
# -------------------------------------------------------------------

# 결과 이미지 나타내기
log_array = log_array.astype(np.uint8)
transformed_image = Image.fromarray(log_array, original_image.mode)

spacing = 10  # 픽셀 단위
total_width = original_image.width * 2 + 3*spacing
total_height = original_image.height + 4*spacing
new_image = Image.new('L', (total_width, total_height), "White")

# 새 캔버스에 이미지 복사
new_image.paste(original_image, (spacing, 3*spacing))  # 원본 이미지를 왼쪽에 붙임
new_image.paste(transformed_image, (original_image.width+2*spacing, 3*spacing))  # 처리된 이미지를 오른쪽에 붙임

# 텍스트 추가
draw = ImageDraw.Draw(new_image)
font_path = r'C:\Windows\Fonts\Arial.ttf'
font_size = 20 
font = ImageFont.truetype(font_path, font_size)

text = "Original"
text_width = getTextSize(draw, text, font)[0]
text_x = (original_image.width - text_width) // 2 + spacing
text_y = 5
draw.text((text_x, text_y), text, font=font, fill="black")

text = "Log Transformation"
text_width = getTextSize(draw, text, font)[0]
text_x = (transformed_image.width - text_width) // 2 + original_image.width+2*spacing
draw.text((text_x, text_y), text, font=font, fill="black")

# 이미지 보기, 저장
new_image.show()
script_name = os.path.basename(__file__)
save_name = os.path.splitext(script_name)[0] + '.png'
new_image.save(f'ch03\\Images\\Result\\{save_name}')
time.sleep(0.5)