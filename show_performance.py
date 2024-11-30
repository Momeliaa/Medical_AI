import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import seaborn as sns

# 저장된 x_test와 y_test 불러오기
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# 성능 지표 계산
# 저장된 model 불러오기
loaded_model = tf.keras.models.load_model('best_global_model.h5')

y_pred = loaded_model.predict(x_test).flatten()  # 예측값을 이진 분류로 변환
y_pred_bin = (y_pred >= 0.5).astype(int)


# Confusion Matrix 계산
cm = confusion_matrix(y_test, y_pred_bin)

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


precision = precision_score(y_test, y_pred_bin)
recall = recall_score(y_test, y_pred_bin)
f1 = f1_score(y_test, y_pred_bin)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# Precision-Recall Curve 그리기
precision_values, recall_values, _ = precision_recall_curve(y_test, y_pred)

plt.plot(recall_values, precision_values, color='blue')
plt.fill_between(recall_values, precision_values, color='blue', alpha=0.2)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()