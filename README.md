augmented_federated_learning.py는 train dataset에서 augmentation을 benign, malignant 둘 다 함
augmented_federated_learning2.py train dataset에서 augmentation을 malignant만 함

=== augmented_federated_learning.py의 client_data 형태 ===
Location: lower extremity
  Train - Benign(0): 512, Malignant(1): 184
  Validation - Benign(0): 32, Malignant(1): 12
  Test - Benign(0): 40, Malignant(1): 15
--------------------------------------------------------------
Location: head/neck
  Train - Benign(0): 512, Malignant(1): 196
  Validation - Benign(0): 32, Malignant(1): 13
  Test - Benign(0): 40, Malignant(1): 16
--------------------------------------------------------------
Location: posterior torso
  Train - Benign(0): 512, Malignant(1): 260
  Validation - Benign(0): 32, Malignant(1): 17
  Test - Benign(0): 40, Malignant(1): 21
--------------------------------------------------------------
Location: anterior torso
  Train - Benign(0): 512, Malignant(1): 208
  Validation - Benign(0): 32, Malignant(1): 13
  Test - Benign(0): 40, Malignant(1): 17
--------------------------------------------------------------
Location: upper extremity
  Train - Benign(0): 512, Malignant(1): 144
  Validation - Benign(0): 32, Malignant(1): 9
  Test - Benign(0): 40, Malignant(1): 12
--------------------------------------------------------------



=== augmented_federated_learning2.py의 client_data 형태 ===
Location: lower extremity
  Train - Benign(0): 512, Malignant(1): 322
  Validation - Benign(0): 128, Malignant(1): 12
  Test - Benign(0): 160, Malignant(1): 15
--------------------------------------------------------------
Location: head/neck
  Train - Benign(0): 511, Malignant(1): 350
  Validation - Benign(0): 129, Malignant(1): 12
  Test - Benign(0): 160, Malignant(1): 16
--------------------------------------------------------------
Location: posterior torso
  Train - Benign(0): 511, Malignant(1): 462
  Validation - Benign(0): 129, Malignant(1): 16
  Test - Benign(0): 160, Malignant(1): 21
--------------------------------------------------------------
Location: anterior torso
  Train - Benign(0): 511, Malignant(1): 371
  Validation - Benign(0): 128, Malignant(1): 13
  Test - Benign(0): 161, Malignant(1): 16
--------------------------------------------------------------
Location: upper extremity
  Train - Benign(0): 511, Malignant(1): 259
  Validation - Benign(0): 128, Malignant(1): 9
  Test - Benign(0): 161, Malignant(1): 11
--------------------------------------------------------------


gradCAM은 작동이 안됨(코드 구현 미흡) -> 해결 필요
