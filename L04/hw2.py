import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

t,bin_img=cv. threshold(img[:,:,3],0,255,cv. THRESH_BINARY+cv. THRESH_OTSU)
img_small=cv.resize(bin_img,dsize=(0,0),fx=0.5,fy=0.5) # 반으로 축소
plt.imshow(bin_img, cmap='gray')
plt.show()

# 5X5 사각형 커널
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

se=np.uint8([[0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]])


# 모폴로지 연산
dilation = cv.morphologyEx(bin_img, cv.MORPH_DILATE, kernel)    #팽창
dilation_small=cv.resize(dilation,dsize=(0,0),fx=0.25,fy=0.25) # 반으로 축소
erosion = cv.morphologyEx(bin_img, cv.MORPH_ERODE, kernel)  #침식
erosion_small=cv.resize(erosion,dsize=(0,0),fx=0.25,fy=0.25) # 반으로 축소
opening = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)  #열기
opening_small=cv.resize(opening,dsize=(0,0),fx=0.25,fy=0.25) # 반으로 축소
closing = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel) #닫기
closing_small=cv.resize(closing,dsize=(0,0),fx=0.25,fy=0.25) # 반으로 축소

stack1 = np.hstack((dilation_small, erosion_small))
stack2 = np.hstack((opening_small, closing_small))
stacked = np.vstack((img_small, stack1, stack2))
plt.imshow(stacked, cmap="gray")
plt.show()
