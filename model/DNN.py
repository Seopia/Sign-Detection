import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import ReLU, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LeakyReLU
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.backend import learning_phase

# CSV 파일 로드
data = pd.read_csv("C:/4-2/deeplearning/project/aiHub/data/data.csv")
print(len(data))

def scale_features(X):
    #scaler = MinMaxScaler() # 0~1 범위로 정규화
    scaler = StandardScaler() #평균 0, 표준편차 1로 정규화
    scaled_X = scaler.fit_transform(X)
    return scaled_X


# 데이터 그룹화 및 분리
def group_and_split(data):
    groups = data.groupby(data['label'])

    train_list = []
    val_list = []
    test_list = []

    for _, group in groups:
        group = group.sample(frac=1).reset_index(drop=True)
        train, temp = train_test_split(group, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.33, random_state=42)  # 0.33 * 0.3 ≈ 0.1

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    train_data = pd.concat(train_list)
    val_data = pd.concat(val_list)
    test_data = pd.concat(test_list)

    return train_data, val_data, test_data


# 스케일링 적용 함수
def split_and_scale_features(X):
    X_pose = scale_features(X.loc[:, X.columns.str.startswith("pose_")].values)
    X_left_hand = scale_features(X.loc[:, X.columns.str.startswith("left_hand_")].values)
    X_right_hand = scale_features(X.loc[:, X.columns.str.startswith("right_hand_")].values)
    X_face = scale_features(X.loc[:, X.columns.str.startswith("face_")].values)
    return [X_pose, X_left_hand, X_right_hand, X_face]


train_data, val_data, test_data = group_and_split(data)

X_train_splits = split_and_scale_features(train_data.iloc[:, 1:-1])
X_val_splits = split_and_scale_features(val_data.iloc[:, 1:-1])
X_test_splits = split_and_scale_features(test_data.iloc[:, 1:-1])

# 라벨 데이터 분리 및 인코딩
y_train = train_data['label'].values
y_val = val_data['label'].values
y_test = test_data['label'].values
#print(f"라벨 수 확인: {len(np.unique(y_train))}")

# 라벨 인코딩
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)



def build_model():
    # 입력 레이어 정의
    pose_input = Input(shape=(50,), name="pose")
    left_hand_input = Input(shape=(42,), name="left_hand")
    right_hand_input = Input(shape=(42,), name="right_hand")
    face_input = Input(shape=(140,), name="face")

    # 각 입력에 대해 별도의 DNN 층 구성
    pose_dense = Dense(128, activation='relu')(pose_input)
    pose_dense = BatchNormalization()(pose_dense)
    pose_dense = Dropout(0.2)(pose_dense)

    left_hand_dense = Dense(64, activation='relu')(left_hand_input)
    left_hand_dense = BatchNormalization()(left_hand_dense)
    left_hand_dense = Dropout(0.2)(left_hand_dense)

    right_hand_dense = Dense(64, activation='relu')(right_hand_input)
    right_hand_dense = BatchNormalization()(right_hand_dense)
    right_hand_dense = Dropout(0.2)(right_hand_dense)

    face_dense = Dense(256, activation='relu')(face_input)
    face_dense = BatchNormalization()(face_dense)
    face_dense = Dropout(0.2)(face_dense)

    # 병합
    merged = Concatenate()([pose_dense, left_hand_dense, right_hand_dense, face_dense])
    merged_dense = Dense(256, activation='relu')(merged)
    merged_dense = BatchNormalization()(merged_dense)
    output = Dense(2604, activation='softmax')(merged_dense)

    # 모델 정의
    model = Model(inputs=[pose_input, left_hand_input, right_hand_input, face_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# 모델 학습
history = model.fit(
    X_train_splits, y_train,
    validation_data=(X_val_splits, y_val),
    epochs=20,
    batch_size=128,
    verbose=1
)

# 모델 평가
train_loss, train_acc = model.evaluate(X_train_splits, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val_splits, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test_splits, y_test, verbose=0)

print(f"Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# 학습 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
