import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 로드 및 그레이스케일 변환환
img=cv.imread('edgeDetectionImage.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 소벨 필터터 적용
sobel_x=cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
sobel_y=cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)

# 에지 강도 계산
edge_strength = cv.magnitude(sobel_x, sobel_y)
edge_strength = cv.convertScaleAbs(edge_strength)  # uint8 변환

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap='gray')
plt.title('Edge Strength')
plt.axis('off')

plt.show()