import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


"""
    학습 및 검증 데이터 준비를 위한 함수
    Args:
        metadata_path (str): 메타데이터 CSV 파일 경로
        image_dir (str): 이미지 디렉토리 경로
        test_size (float): 학습/테스트 데이터 분리 비율
        val_size (float): 학습 데이터에서 검증 데이터 분리 비율
        image_size (tuple): 이미지 크기 (height, width)
        batch_size (int): 배치 크기
    Returns:
        train_dataset (tf.data.Dataset): 학습 데이터셋
        val_dataset (tf.data.Dataset): 검증 데이터셋
        test_dataset (tf.data.Dataset): 테스트 데이터셋
    """

metadata_path = 'C:\\Users\\drago\Desktop\\의료 인공지능\\train-image\\image'
image_dir = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv'


def prepare_data(metadata_path, image_dir,use_fraction, test_size=0.2, val_size=0.2, image_size=(224, 224), batch_size=32):
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['isic_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    # 이미지 파일 존재 여부 필터링
    metadata = metadata[metadata['image_path'].apply(os.path.exists)]
    metadata = metadata.sample(frac=use_fraction, random_state=42).reset_index(drop=True)

    # 이미지와 타겟 변수 추출
    X = metadata['image_path'].tolist()
    y = metadata['target'].tolist()

    # 2. 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42,
                                                      stratify=y_train)

    # 3. 이미지 로드 및 전처리 함수
    def load_and_preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = image / 255.0
        return image, tf.convert_to_tensor(label, dtype=tf.int32)

    # 4. TensorFlow Dataset 생성
    def create_dataset(X, y):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y)).batch(batch_size).shuffle(100)
        return dataset

    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset

metadata_path = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv'
image_dir = 'C:\\Users\\drago\Desktop\\의료 인공지능\\train-image\\image'

train_dataset, val_dataset, test_dataset = prepare_data(
    metadata_path=metadata_path,
    image_dir=image_dir,
    use_fraction=0.1,
    test_size=0.2,
    val_size=0.2,
    image_size=(224, 224),
    batch_size=32
)

# 각 데이터셋의 실제 크기 확인 (배치 크기 제외)
def get_dataset_size_and_target_counts(dataset):
    # 전체 데이터셋에서 타겟별 개수 계산 (배치 크기 영향을 배제)
    targets = []
    for _, label in dataset:
        targets.extend(label.numpy())

    target_counts = pd.Series(targets).value_counts()

    return len(targets), target_counts


# 각 데이터셋 크기 및 타겟별 개수 확인
train_size, train_target_counts = get_dataset_size_and_target_counts(train_dataset)
val_size, val_target_counts = get_dataset_size_and_target_counts(val_dataset)
test_size, test_target_counts = get_dataset_size_and_target_counts(test_dataset)

print(f'Training Dataset Size: {train_size}')
print(f'Training Target Counts: {train_target_counts}')

print(f'Validation Dataset Size: {val_size}')
print(f'Validation Target Counts: {val_target_counts}')

print(f'Test Dataset Size: {test_size}')
print(f'Test Target Counts: {test_target_counts}')