import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

image_dir = 'C:\\Users\\drago\Desktop\\의료 인공지능\\train-image\\image'
metadata_path = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv'
metadata = pd.read_csv(metadata_path)

# 부위별 결측치 제거
metadata = metadata.dropna(subset=['anatom_site_general'])
metadata['image_path'] = metadata['isic_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

tumor_location = metadata['anatom_site_general'].unique()

benign_count=200
epochs=5
batch_size=16

datagen = ImageDataGenerator(
    rotation_range=30,        # 이미지를 최대 30도 회전
    shear_range=0.2,          # 이미지 시어 변환
    zoom_range=0.2,           # 이미지 줌 변환
    horizontal_flip=True,     # 이미지를 좌우 반전
)


# 이미지 로드 함수
def load_images(image_paths):
    images = []
    for i in image_paths:
        img = load_img(i, target_size=(100, 100))

        # [0, 1]로 표준화 후 tensor로 변환
        img_tensor = tf.convert_to_tensor(img_to_array(img) / 255.0, dtype=tf.float32)
        images.append(img_tensor)

    return tf.stack(images)


# augmentation 적용할 때 이미지, 라벨 반환
def load_images_with_labels(image_paths, labels, augmentation_count=3):
    images, augmented_labels = [], []
    for image_path, label in zip(image_paths, labels):
        # 원본 이미지 로드 및 추가
        img = load_img(image_path, target_size=(100, 100))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        augmented_labels.append(label)  # 원본 이미지의 라벨 추가

        # 증강 이미지 생성 및 추가
        img_array = np.expand_dims(img_array, axis=0)
        aug_iter = datagen.flow(img_array, batch_size=1)
        for _ in range(augmentation_count):
            augmented_img = aug_iter.next()[0]
            images.append(augmented_img)
            augmented_labels.append(label)  # 증강된 이미지에 동일한 레이블 추가

    return tf.convert_to_tensor(images, dtype=tf.float32), np.array(augmented_labels)


# train, validation, test로 나누는 함수
def split_data_by_location(location_data):
    # benign 데이터 샘플링
    benign_data = location_data[location_data['target'] == 0].sample(n=benign_count, random_state=42)
    malignant_data = location_data[location_data['target'] == 1]

    # benign과 augmented malignant 데이터 병합
    sampled_data = pd.concat([benign_data, malignant_data])

    # train, validation, test로 나누기
    train_data, test_data = train_test_split(
        sampled_data, test_size=0.2, stratify=sampled_data['target'], random_state=42
    )
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, stratify=train_data['target'], random_state=42
    )

    return train_data, val_data, test_data


clients_data = {}

for location in tumor_location:
    # 부위별 데이터 추출
    location_data = metadata[metadata['anatom_site_general'] == location]
    location_data.loc[:, 'image_path'] = location_data['isic_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    # train, validation, test 데이터 분할
    train_data, val_data, test_data = split_data_by_location(location_data)

    # 증강 데이터 포함한 이미지 및 레이블 로드
    train_images, train_labels = load_images_with_labels(train_data['image_path'], train_data['target'].values)
    val_images, val_labels = load_images(val_data['image_path']), val_data['target'].values
    test_images, test_labels = load_images(test_data['image_path']), test_data['target'].values

    # 데이터 로드
    clients_data[location] = {
        'train': {'images': train_images, 'labels': train_labels},
        'val': {'images': val_images, 'labels': val_labels},
        'test': {'images': test_images, 'labels': test_labels},
    }



# 모든 클라이언트의 테스트, 검증 데이터를 합침
x_val = np.concatenate([data['val']['images'].numpy() for data in clients_data.values()], axis=0)
y_val = np.concatenate([data['val']['labels'] for data in clients_data.values()], axis=0)

x_test = np.concatenate([data['test']['images'].numpy() for data in clients_data.values()], axis=0)
y_test = np.concatenate([data['test']['labels'] for data in clients_data.values()], axis=0)

np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

# 이미지와 라벨 개수가 같은지 확인
for location, data in clients_data.items():
    assert len(data['train']['images']) == len(data['train']['labels']), f"Train data mismatch in {location}"
    assert len(data['val']['images']) == len(data['val']['labels']), f"Validation data mismatch in {location}"
    assert len(data['test']['images']) == len(data['test']['labels']), f"Test data mismatch in {location}"

    print(f"Location: {location}")
    print(f"  Train - Benign(0): {np.sum(data['train']['labels'] == 0)}, Malignant(1): {np.sum(data['train']['labels'] == 1)}")
    print(f"  Validation - Benign(0): {np.sum(data['val']['labels'] == 0)}, Malignant(1): {np.sum(data['val']['labels'] == 1)}")
    print(f"  Test - Benign(0): {np.sum(data['test']['labels'] == 0)}, Malignant(1): {np.sum(data['test']['labels'] == 1)}")
    print("--------------------------------------------------------------")


# ========================================================================================
# 연합 학습

# 각 클라이언트의 모델 생성 함수
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 클라이언트 모델 초기화 함수
def initialize_client_models(num_clients, global_weights=None):
    client_models = []
    for _ in range(num_clients):
        model = create_model()
        if global_weights is not None:
            model.set_weights(global_weights)  # 모든 클라이언트 모델에 동일한 글로벌 가중치 설정
        client_models.append(model)
    return client_models


# FedAvg로 글로벌 모델 업데이트 함수
def federated_averaging(client_models, client_data):
    global_model = create_model()

    # 각 클라이언트의 샘플 수를 가져옴
    num_samples = [data['train']['images'].shape[0] for data in client_data.values()]
    total_samples = sum(num_samples)
    client_weights = [samples / total_samples for samples in num_samples]

    # 각 클라이언트의 가중치를 평균화
    model_weights = [model.get_weights() for model in client_models]
    average_weights = []
    for weights_list in zip(*model_weights):
        average_weights.append(np.average(weights_list, axis=0, weights=client_weights))

    global_model.set_weights(average_weights)
    return global_model



# 각 클라이언트 모델 학습 함수(데이터 개수가 적어 train만 함)
def train_client_model(client_data, location, model):
    history = model.fit(client_data[location]['train']['images'],
                        client_data[location]['train']['labels'],
                        epochs=epochs, batch_size=batch_size, verbose=1)

    return history


# 연합 학습 반복 함수(validation, test)
def federated_learning(rounds, num_clients, client_data, x_test, y_test):
    # 초기 글로벌 모델 설정
    global_model = create_model()
    global_model_accuracy = []
    global_model_loss = []
    global_model_val_accuracy = []
    global_model_val_loss = []

    best_val_loss = float('inf')
    best_model = None  # 성능이 가장 좋은 모델 저장

    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")

        # 클라이언트 모델 초기화
        client_models = initialize_client_models(num_clients, global_weights=global_model.get_weights())

        # 클라이언트 모델 학습
        for i, location in enumerate(client_data.keys()):
            print(f"Training client model {i + 1} at {location}...")
            train_client_model(client_data, location, client_models[i])

        # FedAvg로 글로벌 모델 생성
        global_model = federated_averaging(client_models, client_data)

        # 글로벌 모델 test 평가
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
        global_model_accuracy.append(test_acc * 100)
        global_model_loss.append(test_loss)

        # 글로벌 모델 validation 평가
        val_loss, val_acc = global_model.evaluate(x_val, y_val, verbose=0)
        global_model_val_accuracy.append(val_acc * 100)
        global_model_val_loss.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_global_model = global_model  # 성능이 가장 좋은 모델 저장
            print(f"New best model found! Saving model with val_loss: {val_loss:.2}")

        print(f'Global model test accuracy: {test_acc * 100:.2f}%')

    return best_global_model, global_model_accuracy, global_model_loss, global_model_val_accuracy, global_model_val_loss


# 클라이언트 데이터 준비
num_clients = len(tumor_location)  # 5개의 클라이언트
num_rounds = 7  # 연합학습 7번 수행

# 연합 학습 시작
best_global_model, global_model_accuracy, global_model_loss, global_model_val_accuracy, global_model_val_loss \
    = federated_learning(num_rounds, num_clients, clients_data, x_test, y_test)

best_global_model.save('best_global_model.h5')

# 훈련/검증 정확도 및 손실 그래프 그리기
plt.figure(figsize=(12, 6))

# 정확도 그래프 (훈련 데이터와 검증 데이터)
plt.subplot(1, 2, 1)
plt.plot(range(num_rounds), global_model_accuracy, label='Train Accuracy', color='blue')
plt.plot(range(num_rounds), global_model_val_accuracy, label='Validation Accuracy', color='green')
plt.title('Accuracy Comparison')
plt.xlabel('Rounds')
plt.ylabel('Accuracy (%)')
plt.legend()

# 손실 그래프 (훈련 데이터와 검증 데이터)
plt.subplot(1, 2, 2)
plt.plot(range(num_rounds), global_model_loss, label='Train Loss', color='red')
plt.plot(range(num_rounds), global_model_val_loss, label='Validation Loss', color='orange')
plt.title('Loss Comparison')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
