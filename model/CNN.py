import pandas as pd
import numpy as np
from keras.layers import GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Concatenate, BatchNormalization, ReLU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# 정규화
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

    # 각 단어 그룹 돌면서 train, val, test 나눔
    for _, group in groups:
        group = group.sample(frac=1).reset_index(drop=True)
        train, temp = train_test_split(group, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.33, random_state=42)

        # 리스트에 append
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    # 데이터 프레임 위아래로 합치는 과정
    train_data = pd.concat(train_list)
    val_data = pd.concat(val_list)
    test_data = pd.concat(test_list)

    return train_data, val_data, test_data

# 스케일링, CNN 입력 형태 변환
def split_and_scale_features(X):
    # 특징점 분리
    # loc[:, ...] : 모든 행에 대해 열 이름이 startswith()로 시작하는 값 추출
    X_pose = scale_features(X.loc[:, X.columns.str.startswith("pose_")].values)
    X_left_hand = scale_features(X.loc[:, X.columns.str.startswith("left_hand_")].values)
    X_right_hand = scale_features(X.loc[:, X.columns.str.startswith("right_hand_")].values)
    X_face = scale_features(X.loc[:, X.columns.str.startswith("face_")].values)

    # CNN 입력 형태로 변환 (batch, time_steps, features)
    X_pose = X_pose[..., np.newaxis]  # (batch, 50, 1)
    X_left_hand = X_left_hand[..., np.newaxis]  # (batch, 42, 1)
    X_right_hand = X_right_hand[..., np.newaxis]  # (batch, 42, 1)
    X_face = X_face[..., np.newaxis]  # (batch, 140, 1)

    return [X_pose, X_left_hand, X_right_hand, X_face]


def build_model():
    # 여기서는 특징 추출만
    def cnn(input_shape):
        input_layer = Input(shape=input_shape)
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x) #배치 입력을 정규화(학습 속도 향상 및 과적합 방지용)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x) #시계열 데이터 길이를 평균값으로 변경
        return input_layer, x

    # 각 특징점에 대해 CNN 블록 적용
    pose_input, pose_features = cnn((50, 1))
    left_hand_input, left_hand_features = cnn((42, 1))
    right_hand_input, right_hand_features = cnn((42, 1))
    face_input, face_features = cnn((140, 1))

    # 병합 후 추가 Dense 레이어
    merged = Concatenate()([pose_features, left_hand_features, right_hand_features, face_features]) # 개별 특징을 CNN 작업했으니 각 특징점의 출력 텐서를 하나로 합쳐 다음 레이어로 전달함
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(2604, activation='softmax')(x)

    # 모델 정의
    model = Model(inputs=[pose_input, left_hand_input, right_hand_input, face_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# CSV 파일 로드
data = pd.read_csv("C:/4-2/deeplearning/project/aiHub/data/data.csv")
print("데이터 개수:", len(data))

train_data, val_data, test_data = group_and_split(data)

X_train = split_and_scale_features(train_data.iloc[:, 1:-1])
X_val = split_and_scale_features(val_data.iloc[:, 1:-1])
X_test = split_and_scale_features(test_data.iloc[:, 1:-1])

# 라벨 데이터 분리 및 인코딩
y_train = train_data['label'].values
y_val = val_data['label'].values
y_test = test_data['label'].values

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)


model = build_model()
model.summary()

# EarlyStopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(
    [X_train[0], X_train[1], X_train[2], X_train[3]], y_train,
    validation_data=([X_val[0], X_val[1], X_val[2], X_val[3]], y_val),
    epochs=26,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

# 모델 평가
train_loss, train_acc = model.evaluate(X_train, y_train)
val_loss, val_acc = model.evaluate(X_val, y_val)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Train acc: {train_acc:.4f}, Train loss: {train_loss:.4f}")
print(f"Val acc: {train_acc:.4f}, Val loss: {val_loss:.4f}")
print(f"Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# 시각화(테스트, 검증)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

