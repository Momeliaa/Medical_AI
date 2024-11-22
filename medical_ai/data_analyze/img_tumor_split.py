import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img1 = cv2.imread('C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\malignant\\ISIC_0521019.jpg')
img2 = cv2.imread('C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\benign\\ISIC_0553277.jpg')

# 이미지 크기 조정
resized_img1 = cv2.resize(img1, (224, 224))
resized_img2 = cv2.resize(img2, (224, 224))

def extract_tumor(img1, img2):

    # 이미지를 BGR에서 HSV로 변환
    img1_hsv = cv2.cvtColor(resized_img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(resized_img2, cv2.COLOR_BGR2HSV)

    # 점을 추출할 기준인 Saturation와 Value 범위 설정
    # 높은 채도와 밝기를 가진 부분만 추출
    sat_min, sat_max = 100, 255
    val_min, val_max = 100, 255

    # 각 이미지에서 S와 V 범위로 마스크 생성
    mask1 = cv2.inRange(img1_hsv, (0, sat_min, val_min), (180, sat_max, val_max))
    mask2 = cv2.inRange(img2_hsv, (0, sat_min, val_min), (180, sat_max, val_max))

    # 마스크 적용하여 점 부분만 추출
    extracted_img1 = cv2.bitwise_and(resized_img1, resized_img1, mask=mask1)
    extracted_img2 = cv2.bitwise_and(resized_img2, resized_img2, mask=mask2)

    return mask1, mask2, extracted_img1, extracted_img2



mask1, mask2, extracted_img1, extracted_img2 = extract_tumor(resized_img1, resized_img2)

# extract된 mal, ben 저장
plt.imshow(cv2.cvtColor(extracted_img1, cv2.COLOR_BGR2RGB))
plt.title('Extracted Points - Malignant')
plt.axis('auto')
plt.savefig('Extracted Malignant')
plt.close()

plt.imshow(cv2.cvtColor(extracted_img2, cv2.COLOR_BGR2RGB))
plt.title('Extracted Points - Benign')
plt.axis('auto')
plt.savefig('Extracted Benign')
plt.close()


# 결과 시각화
plt.figure(figsize=(12, 12))

# Original Images
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB))
plt.title('Image 1 (Malignant)')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(cv2.cvtColor(resized_img2, cv2.COLOR_BGR2RGB))
plt.title('Image 2 (Benign)')
plt.axis('off')

# Masks
plt.subplot(3, 2, 3)
plt.imshow(mask1, cmap='gray')
plt.title('Mask - Malignant')
plt.axis('auto')

plt.subplot(3, 2, 4)
plt.imshow(mask2, cmap='gray')
plt.title('Mask - Benign')
plt.axis('auto')

# Extracted Points
plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(extracted_img1, cv2.COLOR_BGR2RGB))
plt.title('Extracted Points - Malignant')
plt.axis('auto')

plt.subplot(3, 2, 6)
plt.imshow(cv2.cvtColor(extracted_img2, cv2.COLOR_BGR2RGB))
plt.title('Extracted Points - Benign')
plt.axis('auto')


plt.tight_layout()
plt.show()
