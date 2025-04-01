import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img1 = cv.imread("mot_color70.jpg")
img2 = cv.imread("mot_color83.jpg")
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# SIFT 특징점 검출기 생성
sift=cv.SIFT_create()
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)
print('특징점 개수:', len(kp1), len(kp2))

# BFMatcher 객체 생성 및 매칭 수행
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭 결과 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과 이미지 그리기
draw_params = dict(flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches = cv.drawMatches(img1_rgb, kp1, img2_rgb, kp2, matches[:50], None, **draw_params)

# 결과 출력
plt.figure(figsize=(12, 6))
plt.imshow(img_matches, cmap='gray')
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()
