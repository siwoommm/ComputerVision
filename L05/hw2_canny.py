import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 및 그레이스케일 변환
img = cv.imread('dabotap.jpg')
img_copy = img.copy()
gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)

# 캐니 에지 검출
edges = cv.Canny(gray, 100, 200)

# 허프 변환을 이용한 직선 검출
lines = cv.HoughLinesP(edges, rho=0.5, theta=np.pi/360, threshold=30, minLineLength=30, maxLineGap=5)


# 검출된 직선을 원본 이미지에 표시
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))
plt.title('Canny & Hough')
plt.axis('off')

plt.show()
