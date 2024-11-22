import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
extracted_mal = cv2.imread('Extracted Malignant.png')
extracted_ben = cv2.imread('Extracted Benign.png')


# HSV 및 그레이스케일 히스토그램 계산 및 시각화
def calculate_and_plot_histograms(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # HSV 히스토그램 계산 (검정색 부분 제외)
    mask = cv2.inRange(hsv_image, (0, 1, 1), (180, 255, 255))

    filtered_h = h[mask > 0]
    filtered_s = s[mask > 0]
    filtered_v = v[mask > 0]

    # Grayscale 히스토그램 계산
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filtered = gray_image[mask > 0]

    return filtered_h, filtered_s, filtered_v, gray_filtered

mal_h, mal_s, mal_v, mal_gray = calculate_and_plot_histograms(extracted_mal)
ben_h, ben_s, ben_v, ben_gray = calculate_and_plot_histograms(extracted_ben)


# Malignant에서 히스토그램 최대값 계산(mal과 ben의 y축 히스토그램을 맞추기 위함)
h_max = np.histogram(mal_h, bins=180)[0].max()
s_max = np.histogram(mal_s, bins=256)[0].max()
v_max = np.histogram(mal_v, bins=256)[0].max()
gray_max = np.histogram(mal_gray, bins=256)[0].max()

# HSV 히스토그램
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.hist(mal_h, bins=180, color='r', alpha=0.7)
plt.title('Hue Histogram (Malignant)')
plt.xlabel('Hue')
plt.ylabel('Frequency')
plt.ylim(0, h_max)

plt.subplot(2, 3, 2)
plt.hist(mal_s, bins=256, range = (0, 256),color='g', alpha=0.7)
plt.title('Saturation Histogram (Malignant)')
plt.xlabel('Saturation')
plt.ylim(0, s_max)

plt.subplot(2, 3, 3)
plt.hist(mal_v, bins=256, range = (0, 256), color='b', alpha=0.7)
plt.title('Value Histogram (Malignant)')
plt.xlabel('Value')
plt.ylim(0, v_max)

plt.subplot(2, 3, 4)
plt.hist(ben_h, bins=180, color='r', alpha=0.7)
plt.title('Hue Histogram (Benign)')
plt.xlabel('Hue')
plt.ylim(0, h_max)

plt.subplot(2, 3, 5)
plt.hist(ben_s, bins=256, range = (0, 256), color='g', alpha=0.7)
plt.title('Saturation Histogram (Benign)')
plt.xlabel('Saturation')
plt.ylim(0, s_max)

plt.subplot(2, 3, 6)
plt.hist(ben_v, bins=256, range = (0, 256), color='b', alpha=0.7)
plt.title('Value Histogram (Benign)')
plt.xlabel('Value')
plt.ylim(0, v_max)

plt.tight_layout()

plt.savefig('hsv_histogram.png')

plt.show()

# Grayscale 히스토그램
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(mal_gray, bins=256, color='gray', alpha=0.7)
plt.title('Grayscale Histogram (Malignant)')
plt.xlabel('Pixel')
plt.ylabel('Frequency')
plt.ylim(0, gray_max)

plt.subplot(1, 2, 2)
plt.hist(ben_gray, bins=256, color='gray', alpha=0.7)
plt.title('Grayscale Histogram (Benign)')
plt.xlabel('Pixel')
plt.ylim(0, gray_max)

plt.tight_layout()

plt.savefig('grayscale_histogram.png')

plt.show()
