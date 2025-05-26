import time
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 모델 및 인코더 로드
model = load_model("C:/Users/mjson/PycharmProjects/sign-detection/model/v4/sign_dnn_model.h5")
label_encoder = joblib.load("C:/Users/mjson/PycharmProjects/sign-detection/model/v4/label_encoder.pkl")

# 더미 데이터 (속도 측정할때는 값이 아닌 배열의 크기와 구조가 중요함)
pose = np.random.rand(1, 50)
left = np.random.rand(1, 42)
right = np.random.rand(1, 42)
face = np.random.rand(1, 140)

model.predict([pose, left, right, face])

# 측정 시작
num_trials = 100
start = time.time()
for _ in range(num_trials):
    model.predict([pose, left, right, face], verbose=0)
end = time.time()

avg_time = (end - start) / num_trials
fps = 1 / avg_time

print(f"평균 처리 시간: {avg_time:.4f}초 | FPS: {fps:.2f}")
