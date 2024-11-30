import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tf_keras_vis.gradcam import Gradcam
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2


# 저장된 이미지 x_test 불러오기
x_test = np.load('x_test.npy')

model = load_model('best_global_model.h5')

model.summary()

# Grad-CAM을 계산하기 위한 함수
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 입력 이미지를 마지막 convolutional layer의 출력과 모델의 예측 결과로 맵핑하는 모델 생성
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 그래디언트를 계산하기 위한 GradientTape 사용
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])  # 예측된 클래스 중 가장 확률이 높은 클래스를 선택
        class_channel = preds[:, pred_index]

    # 마지막 convolutional layer의 feature map에 대한 클래스 출력의 그래디언트 계산
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 각 채널에 대한 그래디언트 평균 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 그래디언트 평균을 feature map과 곱하여 heatmap 생성
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)  # 차원 축소 (불필요한 차원 제거)

    # heatmap을 0과 1 사이로 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 첫 번째 테스트 이미지에 대한 Grad-CAM heatmap 생성
heatmap = make_gradcam_heatmap(x_test[0:1], model, 'conv2d_56')

# 생성된 heatmap 시각화
plt.matshow(heatmap)
plt.colorbar()  # 색상 막대 추가
plt.show()