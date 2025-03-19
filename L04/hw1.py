import cv2 as cv
import matplotlib.pyplot as plt
import sys

# cv.imread()를사용하여이미지를불러옵니다.
img = cv.imread('mistyroad.jpg')
if img is None :
    sys.exit('파일이 존재하지 않습니다.')

# 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환

threshold_value = 127
t,bin_img=cv.threshold(gray[ :,:,2],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

gray_histo=cv.calcHist([gray],[2],None,[256],[0,256])# 2번 채널인 R 채널에서 히스토그램 구함
plt.plot(gray_histo,color='r',linewidth=1)
plt.show()

h=cv.calcHist([img],[2],None,[256],[0,256])# 2번 채널인 R 채널에서 히스토그램 구함
plt.plot(h,color='r',linewidth=1)
plt.show()


cv.imshow('R channel',img[:, :,2])
cv.imshow('R channel binarization',bin_img) # R 채널 이진화 영상

cv.waitKey()
cv.destroyAllWindows ()