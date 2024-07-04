# Digital-Image-Processing
곤잘레스 Digital Image Processing  
예제 python으로 적용해보기  

![diagram](https://github.com/JuHyunLee99/Digital-Image-Processing/assets/123914434/45ec4d14-04ea-421e-b941-1877d496544f)

> ※ 모듈 PIL, numpy, matplotlib.pyplot  
> pyplot
> - 이미지 띄우면 크기 바껴서 나옴. 보간법 등으로 픽셀 달라지고,  plt.imshow(img)할때 cmap (컬러맵), vmin과 vmax 설정 필요.   
> - 그레이스케일 이미지에 'gray' 컬러맵 설정.
> - vmin과 vmax 설정 안하면 컨트라스트 스트레칭되서 띄어짐.
> - > 히스토그램 그릴때만 사용.  (이미지는 PIL로 띄움)
## 1. Image Enhancement
### 1.1 Spatial Domain
- Point Operations (Intensity Transformations)
- Spatial Filters (or Mask, Kernel)
#### 1.1.1 Intensity Transformations
##### 1. Negative
`s = L - 1 - s`
- r : 원본 이미지의 픽셀 값
- s : 변환된 이미지의 픽셀 값
- L : 최대 밝기 레벨의 수 (8비트 이미지 L = 256

``` python
max_value = np.iinfo(original_array.dtype).max
negative_array = max_value - original_array
```
![](ch03/Images/Result/Intensity_Transformations/ex01_Negative.png)

##### 2. Log

`s = c * log(1 + r) for r >= 0`
``` python
original_image = Image.open(image_path)
original_array = np.array(original_image, dtype=np.float32)

c_log = 255 / np.log(1 + np.max(original_array))   # c: 스케일링 상수 =>  표준 8비트 그레이스케일 범위 [0, 255] 벗어나지 않도록 
log_array = c_log * np.log(1 + original_array) # np.log(1 + original_array)에서 256으로 오버플로우 발생하므로 dtype=np.float32로 설정.
```
![](ch03/Images/Result/Intensity_Transformations/ex02_Log.png)

##### 3. Power Low

`s = c * r^γ where c and γ are positive constants`

※ 오프셋  
`s = c * (r + o)^γ where c, γ are positive constants, and o is the offset`

``` python
def gammaTransform(gamma, original_array):
 c_gamma = 255 / np.power(np.max(original_array), gamma)
 gamma_array = c_gamma * np.power(original_array, gamma)
 gamma_array = gamma_array.astype(np.uint8)
 return gamma_array
```

- **Gamma Correction**
   ![](ch03/Images/Result/Intensity_Transformations/ex03_PowerLow.png)
   
- **Constrast Enhancement**
   ![](ch03/Images/Result/Intensity_Transformations/ex04_PowerLow.png)
   ![](ch03/Images/Result/Intensity_Transformations/ex05_PowerLow.png)
  
##### **※ Log VS PowerLow(γ < 1)**
![](ch03/Images/Result/Intensity_Transformations/ex06_LogVsPowerLow_1.png)
Log 변환은 이미지의 밝기 범위를 균등하게 확장하여 이미지에서 밝기 값이 낮은 영역(어두운 부분)의 세부사항이 크게 강조.
로그 변환된 이미지는 패턴과 질감이 더욱 명확하게 드러남.
PowerLow(γ < 1)는  전체적으로 이미지를 부드럽게 하면서도 중요한 영역의 시각적인 구분을 명확하게 하는데 도움.
![](ch03/Images/Result/Intensity_Transformations/ex06_LogVsPowerLow_2.png)
![](ch03/Images/Result/Intensity_Transformations/ex06_LogVsPowerLow_3.png)
로그 변환은 특히 어두운 부분의 디테일을 더욱 선명하게 보이도록 해주는 반면, 파워 로 변환은 밝은 부분의 대비를 더욱 증가시켜줍니다.
##### 4. Piecewise Linear
- **Contrast Stretching**
  
  `(r1, s1) = (r_min, 0), (r2, s2) = (r_max, L-1)`
  ``` python
   def contrastStretching(original_array):
    min_val = np.min(original_array)
    max_val = np.max(original_array)
    # 0 ~ 255 스케일링
    # original_array - min_val :  픽셀 값에서 최소값을 뺌. 데이터의 최소값은 0이 됨.
    # / (max_val - min_val) : 데이터의 범위를 0에서 1사이로 정규화
    # * 255 : 0에서 255 사이의 값으로 확장
    stretched_array = (original_array - min_val) / (max_val - min_val) * 255 
    stretched_array = stretched_array.astype(np.uint8)
    return stretched_array
  ```
  `r1 = r2, s1 = 0, s2 = L - 1`
  ``` python
  def thresholding(original_array):
   avg_val = np.average(original_array)
   # 평균값으로 이진화
   thresholded_array =np.where(original_array >= avg_val, 255, 0)
   thresholded_array = thresholded_array.astype(np.uint8)
   return thresholded_array
  ```
  ![](ch03/Images/Result/Intensity_Transformations/ex07_PiecewiseLinear.png)

  
- **Inensity-Level-Slicing**
  
  특정 밝기 범위 강조
  ``` python
   def intensityLevelSlicing(original_array, lower, upper, binary_mode):
    
    if binary_mode:
        sliced_array = np.where((original_array >= lower) & (original_array <= upper), 255, 0)
    else:
        sliced_array = np.where((original_array >= lower) & (original_array <= upper), 0, original_array)
        
    sliced_array = sliced_array.astype(np.uint8)
    return sliced_array
  ```
  ![](ch03/Images/Result/Intensity_Transformations/ex08_PiecewiseLinear.png) 
  범위는 임의로 했음. 뒤로 가면 범위도 어떻게 정해야하는지 나오기 않을까...
  
- **Bit-plane slicing**
  
  특정 비트의 기여를 강조
  ``` python
  def bitPlaneSlicing(original_array, i):
      mask = 1 << i
      sliced_array = (np.bitwise_and(original_array, mask) >> i)*255       
      sliced_array = sliced_array.astype(np.uint8)
      return sliced_array
  ```
  ![](ch03/Images/Result/Intensity_Transformations/ex09_PiecewiseLinear.png)

  ``` python
  def bitPlaneSlicing(original_array, i):
    mask = 1 << i
    sliced_array = (np.bitwise_and(original_array, mask) >> i)     
    sliced_array = sliced_array.astype(np.uint8)
    return sliced_array

  def reconstructing(original_array, Indexs):
      sliced_array = np.zeros_like(original_array)
      for num in Indexs:
          sliced_array += bitPlaneSlicing(original_array, num) * 2**num
      return sliced_array
  ```
  ![](ch03/Images/Result/Intensity_Transformations/ex10_PiecewiseLinear.png)
  영상 압축에 유용
  네 개의 최상위 비트 평면들을 저장하면 만족할 만한 디테일을 갖게 원래 영상을 복구할 수 있다.

#### 1.1.2 Histogram Processing

`h(r_k) = n_k  where 0 ≤ r_k ≤ L-1`
- r_k : k번째 밝기 값
- n_k : 영상에서 밝기 r_k를 갖는 화소들 수

정규화  
`p(r_k) = n_k/NM`  
- N, M = 영상의 행과 열 수  
=> p(r_k)는 영상에서 밝기 레벨 r_k가 발생할 확률의 추정  
=> 정규화된 히스토그램의 모든 요소의 합은 1
![0](ch03/Images/Result/Histogram_Processing/ex01_Histogram.png)
