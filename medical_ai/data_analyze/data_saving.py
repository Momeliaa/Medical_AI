from sklearn.model_selection import train_test_split
import pandas as pd
import os

def prepare_data(metadata_path, image_dir, use_fraction, test_size=0.2, val_size=0.2):
    # 메타데이터 읽기
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['isic_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    # 이미지 파일 존재 여부 필터링
    metadata = metadata[metadata['image_path'].apply(os.path.exists)]
    metadata = metadata.sample(frac=use_fraction, random_state=42).reset_index(drop=True)

    # 이미지 경로와 타겟 변수 추출
    X = metadata['image_path'].tolist()
    y = metadata['target'].tolist()

    # train, test 분할 (stratify=y로 타겟 비율 유지)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # train, val 분할 (stratify=y_train로 타겟 비율 유지)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42, stratify=y_train)

    # 각 데이터셋 반환
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    return train_data, val_data, test_data

# 파일 경로 설정
metadata_path = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-metadata.csv'
image_dir = 'C:\\Users\\drago\\Desktop\\의료 인공지능\\train-image\\image'

# 데이터 준비 (train, val, test 분할)
train_data, val_data, test_data = prepare_data(
    metadata_path=metadata_path,
    image_dir=image_dir,
    use_fraction=1.0,  # 전체 데이터 사용
    test_size=0.2,  # 테스트 데이터 비율
    val_size=0.2   # 검증 데이터 비율
)

# 각 데이터셋의 크기 및 타겟별 개수 확인
train_target_counts = pd.Series([label for _, label in train_data]).value_counts()
val_target_counts = pd.Series([label for _, label in val_data]).value_counts()
test_target_counts = pd.Series([label for _, label in test_data]).value_counts()

print(f'Training Target Counts: {train_target_counts}')
print(f'Validation Target Counts: {val_target_counts}')
print(f'Test Target Counts: {test_target_counts}')
