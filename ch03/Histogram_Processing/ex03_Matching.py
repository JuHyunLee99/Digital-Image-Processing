from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import time
import io

def histogram_equalizatioin(origin_array):
    # 히스토그램 계산
    hist, bins = np.histogram(origin_array, 256, [0,256])
    hist_normalized = hist / hist.sum()
    
    # 누적 분포 함수(CDF) 계산
    cdf = hist_normalized.cumsum()
    # 평활화, Round
    cdf_scaled = np.round(cdf * 255).astype(np.uint8)
    
    # ※ 다차원 배열을 1차원으로 만들기
    # flatten은 복사본을 생성 
    # ravel은 참조
    
    # 원본 배열의 픽셀 값에 CDF를 매핑
    equalized_array = cdf_scaled[origin_array.flatten()].reshape(origin_array.shape)    
    
    return (equalized_array, cdf_scaled)

def getHistImg(array):
    # 플롯 설정
    plt.figure(figsize=(8, 6))
    plt.hist(array.ravel(), bins=256, color='black',density=True)
    plt.xlabel('r')
    plt.ylabel('p(r)')
    plt.xlim(-1, 257)  
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    buf.seek(0)
    return Image.open(buf)

def getFuncImg(array):
    plt.figure(figsize=(8, 6))
    plt.plot(array)
    plt.xlabel('r')
    plt.ylabel('s')
    plt.xlim(0, 256)  
    plt.ylim(0, 256)  
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    buf.seek(0)
    return Image.open(buf)

# 텍스트 바운딩 박스 계산
def getTextSize(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return (text_width, text_height)    

def showImages(images, rows, cols, name, title=True , title_size=40, text_size=30):
    
    # 폰트 설정
    title_font = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', title_size)
    text_font = ImageFont.truetype(r'C:\Windows\Fonts\Arial.ttf', text_size)
    
    spacing = 10  # 텍스트와 이미지 사이의 간격
    
    # 새 캔버스 
    max_widths = [0] * cols
    max_heights = [0] * rows
    
    for idx, img in enumerate(images.values()):       
        row = idx // cols
        col = idx % cols
        
        width, height = img.size
        max_widths[col] = max(max_widths[col], width)
        max_heights[row] = max(max_heights[row], height + text_size + spacing)  # 텍스트 영역 포함
    
    canvas_width = sum(max_widths) + (cols + 1) * spacing
    canvas_height = sum(max_heights) + (rows + 1) * spacing + (title_size + spacing if title else 0)
    canvas = Image.new('L', (canvas_width, canvas_height), 'white')
    
    # 타이틀 그리기 (있을 경우)
    draw = ImageDraw.Draw(canvas)
    if title:
        title_width, _ = getTextSize(draw, fileName, font=title_font)
        title_x = (canvas_width - title_width) // 2
        title_y = spacing
        draw.text((title_x, title_y), fileName, font=title_font, fill="black")
        y_offset = title_y + title_size + spacing
    else:
        y_offset = spacing

    # 이미지와 각 이미지의 텍스트 그리기
    for row in range(rows):
        x_offset = spacing
        for col in range(cols):
            idx = row * cols + col
            if idx < len(images):
                imgName = list(images.keys())[idx]
                img = images[imgName]
                
                # 이미지 이름 텍스트 위치 계산 및 그리기
                text_width, text_height = getTextSize(draw, imgName, font=text_font)
                text_x = x_offset + (max_widths[col] - text_width) // 2
                text_y = y_offset + (max_heights[row] - (img.height + text_size + spacing)) // 2
                draw.text((text_x, text_y), imgName, font=text_font, fill="black")
                
                # 이미지
                img_x = x_offset + (max_widths[col] - img.width) // 2  
                img_y = text_y + text_size + spacing
                canvas.paste(img, (img_x, img_y))

            x_offset += max_widths[col] + spacing
        y_offset += max_heights[row]+ spacing
    
    global file_prefix
    fileName = file_prefix + '_' + name + '.png'      
    canvas.save(rf'C:\Source\Digital-Image-Processing\ch03\Images\Result\Histogram_Processing\{fileName}')
    canvas.show()
    time.sleep(0.5)   

def histogram_matching(array):
    
    return 
    
    
if __name__ == "__main__":
    
    file_prefix = "ex03"
    image_paths = r'C:\Source\Digital-Image-Processing\ch03\Images\Source\Fig0323(a)(mars_moon_phobos).tif'
    origin_image = Image.open(image_paths)
    origin_array = np.array(origin_image)
    
    images = { 
              "Original Image" : origin_image,
              "Histogram": getHistImg(origin_array)
             }
    showImages(images, 1, 2, "Original", False)
    
    equalizated_array, cdf_scaled = histogram_equalizatioin(origin_array)
    
    images = {
                "Equalizated Image": Image.fromarray(equalizated_array),
                "Histogram" : getHistImg(equalizated_array),
                "Transformation Func" : getFuncImg(cdf_scaled)
             }
    showImages(images, 1, 3,"Equalization", False)
    
    