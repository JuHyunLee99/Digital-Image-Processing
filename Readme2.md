##### d. Piecewise Linear
- **Contrast Stretching**  
   ![1](ch03/Images/Result/ex07_PiecewiseLinear.png)
  
   $(r_1, s_1) = (r_min, 0), (r_2, s_2) = (r_max, L-1)$
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
   
   $r_1 = r_2, \quad s_1 = 0, \quad s_2 = L - 1$   
   ``` python
   def thresholding(original_array):
    avg_val = np.average(original_array)
    # 평균값으로 이진화
    thresholded_array =np.where(original_array >= avg_val, 255, 0)
    thresholded_array = thresholded_array.astype(np.uint8)
    return thresholded_array
   ```
   
   
- **Inensity-Level-Slicing**
   ``` python
   def intensityLevelSlicing(original_array, lower, upper, binary_mode):
    
    if binary_mode:
        sliced_array = np.where((original_array >= lower) & (original_array <= upper), 255, 0)
    else:
        sliced_array = np.where((original_array >= lower) & (original_array <= upper), 0, original_array)
        
    sliced_array = sliced_array.astype(np.uint8)
    return sliced_array
   ```
     ![2](ch03\Images\Result\ex08_PiecewiseLinear.png)
