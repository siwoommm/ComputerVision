import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('coffee_cup.jpg')

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
iterCount = 1
mode = cv.GC_INIT_WITH_RECT

# 초기 사각형 영역 설정 (x, y, width, height)
rc = (150, 150, 1000, 700)

cv.grabCut(img, mask, rc, bgdModel, fgdModel, iterCount, mode)

# 마스크 변환 (배경: 0, 객체: 1)
mask_bin = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
dst = img * mask_bin[:, :, np.newaxis]

# 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_bin, cmap='gray')
plt.title('Mask Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.title('Background Removed')
plt.axis('off')

plt.show()
