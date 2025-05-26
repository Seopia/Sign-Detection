# 전체 구조
# 웹캠 + 미디어파이프 -> 좌표 추출 -> POST 전송
# Flask 에서 데이터 받음 -> 모델 호출 -> 예측 -> 예측 결과 반환
# {
#     "pose": [50개 값],
#     "left_hand": [42개 값],
#     "right_hand": [42개 값],
#     "face": [140개 값]
# }
import joblib
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.python.keras.saving.save import load_model

app = Flask(__name__)

# 모델 로드
model = load_model("model/v4/sign_dnn_model.h5")

# 학습 시 저장해둔 label encoder 불러오기
# inverse_transform()을 사용하기 위해 객체 사용
label_encoder = joblib.load("model/v4/label_encoder.pkl")

# 에측
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # 데이터 받아온 것
        pose = np.array(data['pose']).reshape(1, -1)
        left_hand = np.array(data['left_hand']).reshape(1, -1)
        right_hand = np.array(data['right_hand']).reshape(1, -1)
        face = np.array(data['face']).reshape(1, -1)

        # 예측값 전달
        predict = model.predict([pose, left_hand, right_hand, face])
        confidence = np.max(predict) #가장 높은 확률값
        pred_index = np.argmax(predict) #가장 확률이 높은 클래스 번호 반환


        if confidence >= 0.8:
            word = label_encoder.inverse_transform([pred_index])[0]
            return jsonify({'prediction': word, 'confidence': float(confidence)})
        else:
            return jsonify({'prediction': None, 'confidence': float(confidence)})

    except Exception as e:
        return jsonify({'error': str(e)})


#설정
if __name__ == '__main__':
    app.run(debug=True)

