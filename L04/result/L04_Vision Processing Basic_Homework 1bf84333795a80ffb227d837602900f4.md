# L04_Vision Processing Basic_Homework

## 01 **이진화 및 히스토그램 구하기**

### 요구사항

- `cv.imread()`를 사용하여 이미지를 불러오기
- `cv.cvtColor()`를 사용하여 그레이스케일 변환
- `cv.threshold()`를 사용하여 임계값 127을 기준으로 이진화
- `cv.calcHist()`를 사용하여 히스토그램을 계산
- `matplotlib.pyplot`을 사용하여 히스토그램 시각화

### 코드

```python
import cv2 as cv
import matplotlib.pyplot as plt
import sys

# 1. 이미지 로드
img = cv.imread('mistyroad.jpg')
if img is None :
    sys.exit('파일이 존재하지 않습니다.')

# 2. 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환

# 3. 이진화 적용
threshold_value = 127 # 임계값
_, binary = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY)

# 4. 히스토그램 계산
hist_gray = cv.calcHist([gray], [0], None, [256], [0, 256])
hist_binary = cv.calcHist([binary], [0], None, [256], [0, 256])

# 5. 결과 출력
cv.imshow('Original Image', img)
cv.imshow('Grayscale Image', gray)
cv.imshow('Binary Image', binary)

# 5. 히스토그램 출력
plt.figure(figsize=(12, 5))

# (1) 그레이스케일 히스토그램
plt.subplot(1, 2, 1)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(hist_gray, color='black')
plt.xlim([0, 256])

# (2) 이진화 히스토그램
plt.subplot(1, 2, 2)
plt.title("Binary Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(hist_binary, color='red')
plt.xlim([0, 256])

plt.show()
```

### 코드 설명

### (1) 라이브러리 불러오기

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
```

OpenCV(`cv2`)는 이미지 처리, Numpy(`numpy`)는 배열 연산, Matplotlib(`matplotlib.pyplot`)는 데이터 시각화를 위해 사용한다.

### (2) 이미지 불러오기 및 확인

```python
img = cv.imread('image.png')
if img is None:
    print("파일이 존재하지 않습니다.")
    exit()
```

파일이 존재하지 않을 경우 프로그램이 종료되도록 설정하였다.

### (3) 그레이스케일 변환

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

이미지를 `cv.COLOR_BGR2GRAY`를 이용하여 1채널(흑백)로 변환한다.

- `cv.cvtColor(src, code)`
    - `src`: 입력 이미지
    - `code`: 색상 변환 코드 (`cv.COLOR_BGR2GRAY`는 BGR 이미지를 그레이스케일로 변환)

### (4) 이진화 적용

```python
_, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
```

- `cv.threshold(src, thresh, maxval, type)`
    - `src`: 입력 이미지 (그레이스케일이어야 함)
    - `thresh`: 임계값 (127)
    - `maxval`: 임계값을 초과하는 픽셀에 설정할 값 (255)
    - `type`: 임계값 적용 방법 (`cv.THRESH_BINARY`는 기준값보다 크면 `maxval`, 작으면 0)

### (5) 히스토그램 계산

```python
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
binary_hist = cv.calcHist([binary], [0], None, [256], [0, 256])
```

그레이스케일 및 이진화된 이미지의 픽셀 값 분포를 `cv.calcHist()`로 계산한다.

- `cv.calcHist(images, channels, mask, histSize, ranges)`
    - `images`: 입력 이미지 리스트
    - `channels`: 분석할 채널 (그레이스케일이므로 `[0]`)
    - `mask`: 특정 영역만 분석할 경우 사용 (`None`이면 전체 이미지 대상)
    - `histSize`: 히스토그램 빈 개수 (256)
    - `ranges`: 픽셀 값 범위 ([0, 256])

### (6) 이미지 출력

```python
cv.imshow('Original Image', img)
cv.imshow('Grayscale Image', gray)
cv.imshow('Binary Image', binary)
```

OpenCV 창을 이용하여 원본, 그레이스케일, 이진화 이미지를 화면에 출력한다.

### (7) 히스토그램 시각화

```python
plt.figure(figsize=(12, 5))
```

출력 크기를 설정한다.

### (7-1) 그레이스케일 히스토그램

```
plt.subplot(1, 2, 1)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(gray_hist, color='black')
plt.xlim([0, 256])
```

그레이스케일 이미지의 픽셀 값 빈도 분포를 검은색으로 출력한다.

### (7-2) 이진화 히스토그램

```python
plt.subplot(1, 2, 2)
plt.title("Binary Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(binary_hist, color='red')
plt.xlim([0, 256])
```

이진화된 이미지의 히스토그램을 붉은색으로 출력하여 0(검정)과 255(흰색) 픽셀만 포함된 것을 확인할 수 있다.

```
plt.show()
```

모든 그래프를 출력한다.

### 실행 결과

![Original Image](image.png)

Original Image

![Grayscale Image](image%201.png)

Grayscale Image

![Binary Image](image%202.png)

Binary Image

![image.png](image%203.png)

- **그레이스케일 히스토그램**: 픽셀 값(0~255)의 빈도가 연속적으로 분포

- **이진화 히스토그램**: 0과 255 값에서만 픽셀 개수가 집중됨
    - 이는 이진화가 픽셀을 두 가지 값(검은색과 흰색)으로만 변환했기 때문

## 02 **모폴로지 연산 적용하기**

### 요구사항

- `cv.getStructuringElement()`를 사용하여 5x5 사각형 커널 생성
- `cv.morphologyEx()`를 사용하여 팽창, 침식, 열기, 닫기 연산 적용
- `np.hstack()`을 사용하여 원본 이미지와 모폴로지 연산 결과를 나란히 배치
- 결과 이미지를 하나의 화면에 출력

### 코드

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

# 2. 이진화
_, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 3. 이미지 크기 축소
img_small = cv.resize(bin_img, dsize=(0, 0), fx=0.5, fy=0.5)  # 반으로 축소

# 4. 5x5 사각형 커널 생성
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

# 5. 모폴로지 연산
dilation = cv.morphologyEx(bin_img, cv.MORPH_DILATE, kernel)   # 팽창
erosion = cv.morphologyEx(bin_img, cv.MORPH_ERODE, kernel)     # 침식
opening = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)      # 열기
closing = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel)     # 닫기

# 6. 축소된 이미지 버전 생성 (보기 쉽게 축소)
dilation_small = cv.resize(dilation, dsize=(0, 0), fx=0.25, fy=0.25)
erosion_small = cv.resize(erosion, dsize=(0, 0), fx=0.25, fy=0.25)
opening_small = cv.resize(opening, dsize=(0, 0), fx=0.25, fy=0.25)
closing_small = cv.resize(closing, dsize=(0, 0), fx=0.25, fy=0.25)

# 7. 결과 이미지 결합 및 출력
stack1 = np.hstack((dilation_small, erosion_small))
stack2 = np.hstack((opening_small, closing_small))
stacked = np.vstack((img_small, stack1, stack2))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(stacked, cmap='gray')  # 흑백 이미지로 표시
ax.axis('off') 
plt.title("Morphological Operations", fontsize=16)
plt.tight_layout() 
plt.show()

```

### 코드 설명

### (1) 라이브러리 불러오기

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
```

- `cv2`는 OpenCV 라이브러리로 이미지 처리에 사용된다.
- `numpy`는 배열 연산에 사용된다.
- `matplotlib.pyplot`은 이미지 시각화를 위해 사용된다.

### (2) 이미지 불러오기 및 이진화

```python
img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)
_, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```

- `cv.imread()`를 사용해 이미지 파일을 불러옵니다. 알파 채널을 포함하여 불러온다.
- `cv.threshold()`는 Otsu의 이진화 방법을 사용하여 이미지를 이진화한다.

### (3) 이미지 축소

```python
img_small = cv.resize(bin_img, dsize=(0, 0), fx=0.5, fy=0.
```

- `cv.resize()`를 사용하여 이미지를 50% 크기로 축소한다.

### (4) 사각형 커널 생성

```python
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
```

- `cv.getStructuringElement()`를 사용하여 5x5 크기의 사각형 커널을 생성한다.

### (5) 모폴로지 연산

```python
dilation = cv.morphologyEx(bin_img, cv.MORPH_DILATE, kernel)   # 팽창
erosion = cv.morphologyEx(bin_img, cv.MORPH_ERODE, kernel)     # 침식
opening = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)      # 열기
closing = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel)     # 닫기
```

- `cv.morphologyEx()` 함수는 모폴로지 연산을 적용하는 함수.
- `cv.MORPH_DILATE`: 팽창 연산
- `cv.MORPH_ERODE`: 침식 연산
- `cv.MORPH_OPEN`: 열기 연산 (침식 후 팽창)
- `cv.MORPH_CLOSE`: 닫기 연산 (팽창 후 침식)

### (6) 축소된 이미지 버전 생성

```python
dilation_small = cv.resize(dilation, dsize=(0, 0), fx=0.25, fy=0.25)
erosion_small = cv.resize(erosion, dsize=(0, 0), fx=0.25, fy=0.25)
opening_small = cv.resize(opening, dsize=(0, 0), fx=0.25, fy=0.25)
closing_small = cv.resize(closing, dsize=(0, 0), fx=0.25, fy=0.25)
```

- 결과 이미지를 보기 쉽게 25%로 축소한니다.

### (7) 결과 이미지 결합 및 출력

```python
stack1 = np.hstack((dilation_small, erosion_small))
stack2 = np.hstack((opening_small, closing_small))
stacked = np.vstack((img_small, stack1, stack2))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(stacked, cmap='gray')  # 흑백 이미지로 표시
ax.axis('off')
plt.title("Morphological Operations", fontsize=16)
plt.tight_layout()
plt.show()
```

- `np.hstack()`은 이미지를 수평으로 연결하고, `np.vstack()`은 수직으로 연결하여 최종 이미지를 만든다.
- `matplotlib`을 사용하여 이미지를 출력.

### 실행 결과

![image.png](image%204.png)

- **팽창 (Dilation)**: 이미지의 밝은 부분이 확장되어 보인다.
- **침식 (Erosion)**: 이미지의 밝은 부분이 축소되어 보인다.
- **열기 (Opening)**: 작은 잡음을 제거한 후, 이미지가 더 깔끔하게 만든다.
- **닫기 (Closing)**: 작은 구멍이 메워지고, 경계가 부드러워진다.

## 03 기하 연산 및 선형 보간 적용하기

### 요구사항

- `cv.getRotationMatrix2D()`를 사용하여 회전 변환 행렬을 생성한다.
- `cv.warpAffine()`을 사용하여 이미지를 회전 및 확대한다.
- `cv.INTER_LINEAR`을 사용하여 선형 보간을 적용한다.
- 원본 이미지와 회전 및 확대된 이미지를 한 화면에서 비교한다.

### 코드 설명

### (1) 라이브러리 불러오기

```python
import cv2 as cv
import sys
import numpy as np
```

- `cv2`는 OpenCV 라이브러리로, 이미지 처리에 사용된다.
- `numpy`는 배열 연산을 수행하기 위해 사용된다.
- `sys.exit()`는 이미지 파일이 존재하지 않을 경우 프로그램을 종료하기 위해 사용된다.

### (2) 이미지 불러오기 및 크기 조정

```python
img = cv.imread('rose.png')
if img is None:
    sys.exit('파일이 존재하지 않는다.')

img = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)  # 반으로 축소
```

- `cv.imread()`를 사용하여 이미지를 불러온다.
- `cv.resize()`를 사용하여 이미지를 50% 크기로 축소한다.

### (3) 회전 변환 행렬 생성

```python
rows, cols = img.shape[:2]
center = (cols / 2, rows / 2)
rotation_matrix = cv.getRotationMatrix2D(center, 45, 1.5)
```

- `img.shape[:2]`를 사용하여 이미지의 높이와 너비를 가져온다.
- `(cols / 2, rows / 2)`를 사용하여 회전 중심을 이미지의 중앙으로 설정한다.
- `cv.getRotationMatrix2D()`를 사용하여 45도 회전 및 1.5배 확대하는 변환 행렬을 생성한다.

### (4) 변환 적용 (선형 보간 적용)

```
rotated_scaled_img = cv.warpAffine(img, rotation_matrix, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)
```

- `cv.warpAffine()`을 사용하여 이미지를 변환한다.
- `flags=cv.INTER_LINEAR`을 사용하여 선형 보간을 적용한다.
- 출력 이미지 크기는 `(int(cols * 1.5), int(rows * 1.5))`로 설정하여 확대 효과를 적용한다.

### (5) 원본 및 변환된 이미지 출력

```python
cv.imshow('Original Image', img)
cv.imshow('Rotated & Scaled Image', rotated_scaled_img)

cv.waitKey(0)
cv.destroyAllWindows()
```

- `cv.imshow()`를 사용하여 원본 이미지와 변환된 이미지를 각각 출력한다.
- `cv.waitKey(0)`를 사용하여 키 입력을 대기한다.
- `cv.destroyAllWindows()`를 사용하여 모든 창을 닫는다.

### 실행 결과

![image.png](image%205.png)

![image.png](image%206.png)

- 변환된 이미지는 45도 회전되고 1.5배 확대되어 출력된다.
- 선형 보간이 적용되어 변환된 이미지가 부드럽게 표현된다.