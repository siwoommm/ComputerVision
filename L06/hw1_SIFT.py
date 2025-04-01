import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('mot_color70.jpg')    # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# SIFT 객체 생성 (특징점 개수 제한 가능)
sift=cv.SIFT_create(nfeatures=700)
kp,des=sift.detectAndCompute(gray,None)

# 특징점 시각화
gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift', gray)

# 결과 출력
plt.figure(figsize=(10, 6))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))   #BGR 형식이므로 RGB로 변환
plt.title("Original Image")
plt.axis("off")

# 특징점 시각화 이미지
plt.subplot(1, 2, 2)
plt.imshow(gray)
plt.title("SIFT Keypoints")
plt.axis("off")

plt.show()

