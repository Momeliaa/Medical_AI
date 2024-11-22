import cv2
import matplotlib.pyplot as plt

img1_path = 'C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\malignant\\ISIC_0521019.jpg'
img2_path = 'C:\\Users\\drago\\PycharmProjects\\ai_medical\\Dataset\\train\\benign\\ISIC_0553277.jpg'
image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# 히트맵 생성
heatmap_image1 = cv2.applyColorMap(image1, cv2.COLORMAP_JET)
heatmap_image2 = cv2.applyColorMap(image2, cv2.COLORMAP_JET)

# 히트맵 시각화
plt.figure(figsize=(8, 6))

plt.subplot(2, 3, 1)
plt.title("Heatmap of Malignant")
plt.imshow(cv2.cvtColor(heatmap_image1, cv2.COLOR_BGR2RGB))
plt.axis("auto")

plt.subplot(2, 3, 2)
plt.title("Heatmap of Benign")
plt.imshow(cv2.cvtColor(heatmap_image2, cv2.COLOR_BGR2RGB))
plt.axis("auto")

plt.tight_layout()
plt.show()