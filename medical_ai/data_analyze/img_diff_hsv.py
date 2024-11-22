import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img1 = cv2.imread('C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\malignant\\ISIC_4308153.jpg')
img2 = cv2.imread('C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\benign\\ISIC_2005828.jpg')

# 두 이미지의 크기가 다름
print(img1.shape)
print(img2.shape)

# 이미지 크기 조절
resized_img1 = cv2.resize(img1, (224, 224))
resized_img2 = cv2.resize(img2, (224, 224))

# 이미지를 BGR에서 HSV로 변환
img1_hsv = cv2.cvtColor(resized_img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2HSV)

# HSV 채널 분리 (H: Hue, S: Saturation, V: Value)
hue1, sat1, val1 = cv2.split(img1_hsv)
hue2, sat2, val2 = cv2.split(img2_hsv)


plt.figure(figsize=(10, 12))

# 원본 Resized Images
plt.subplot(4, 2, 1)
plt.imshow(cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환하여 표시
plt.title('Image 1(Malignant)')
plt.axis('auto')

plt.subplot(4, 2, 2)
plt.imshow(cv2.cvtColor(resized_img2, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환하여 표시
plt.title('Image 2(Benign)')
plt.axis('auto')

# Hue(색조) Channel
plt.subplot(4, 2, 3)
plt.imshow(hue1, cmap='hsv')
plt.title('Hue - Malignant')
plt.axis('auto')

plt.subplot(4, 2, 4)
plt.imshow(hue2, cmap='hsv')
plt.title('Hue - Benign')
plt.axis('auto')

# Saturation(채도) Channel
plt.subplot(4, 2, 5)
plt.imshow(sat1, cmap='gray')
plt.title('Saturation - Malignant')
plt.axis('auto')

plt.subplot(4, 2, 6)
plt.imshow(sat2, cmap='gray')
plt.title('Saturation - Benign')
plt.axis('auto')

# Value(명도) Channel
plt.subplot(4, 2, 7)
plt.imshow(val1, cmap='gray')
plt.title('Value - Malignant')
plt.axis('auto')

plt.subplot(4, 2, 8)
plt.imshow(val2, cmap='gray')
plt.title('Value - Benign')
plt.axis('auto')

plt.tight_layout()

plt.savefig('hsv_img_visualization_2')

plt.show()
