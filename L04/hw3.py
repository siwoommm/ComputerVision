import cv2 as cv
import sys
import numpy as np

# 1. 이미지 로드
img = cv.imread('rose.png')
if img is None :
    sys.exit('파일이 존재하지 않습니다.')

img=cv.resize(img,dsize=(0,0),fx=0.5,fy=0.5) # 반으로 축소

# 2. 회전 변환 행렬 생성 (45도 회전)
rows, cols = img.shape[:2]
center = (cols / 2, rows / 2)
rotation_matrix = cv.getRotationMatrix2D(center, 45, 1.5)  # 45도 회전 및 1.5배 확대

# 3. cv.warpAffine()을 사용하여 변환 적용
rotated_scaled_img = cv.warpAffine(img, rotation_matrix, (int(cols * 1.5), int(rows * 1.5)), flags=cv.INTER_LINEAR)

# 4. 원본 및 변환된 이미지 비교 출력
cv.imshow('Original Image', img)
cv.imshow('Rotated & Scaled Image', rotated_scaled_img)

cv.waitKey(0)
cv.destroyAllWindows()
