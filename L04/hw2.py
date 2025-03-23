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
