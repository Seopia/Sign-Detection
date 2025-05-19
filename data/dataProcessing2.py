import os
import json
import pandas as pd
import re
from collections import Counter

# 경로 설정
input_csv = r"C:\4-2\deeplearning\project\aiHub\data\data_no_label.csv"
output_csv = r"C:\4-2\deeplearning\project\aiHub\data\data_labeled.csv"
morpheme_base_dir = r"C:\4-2\deeplearning\project\aiHub\data\morpheme\17"

# CSV 파일 로드
df = pd.read_csv(input_csv)

# 라벨 추가를 위한 새로운 컬럼 (마지막 열에 추가)
df['label'] = 'unknown'

# 라벨 캐시 사전
label_cache = {}

# JSON 파일에서 라벨 추출 함수
def extract_label(word_id):
    # 캐시에 이미 있다면 바로 반환
    if word_id in label_cache:
        return label_cache[word_id]

    # JSON 파일 경로 (D, F, B, L, R 순으로 시도)
    for angle in ['D', 'F', 'L', 'R', 'U']:
        json_file = f"NIA_SL_{word_id}_REAL17_{angle}_morpheme.json"
        json_path = os.path.join(morpheme_base_dir, json_file)

        try:
            if os.path.exists(json_path):
                # JSON 파일 열기
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # name 값 추출
                    name = data.get("data", [{}])[0].get("attributes", [{}])[0].get("name", "unknown")
                    # 캐시에 저장
                    label_cache[word_id] = name
                    return name
        except Exception as e:
            print(f"Error reading {json_path}: {str(e)}")

    # 라벨을 찾지 못한 경우
    return "unknown"

# CSV 파일 행 순회하여 라벨 추가
for index, row in df.iterrows():
    try:
        # 파일명에서 WORD ID 추출 (정규표현식 사용)
        file_name = row['file_name']
        match = re.search(r'WORD(\d+)', file_name)
        if match:
            word_id = f"WORD{match.group(1).zfill(4)}"
            label = extract_label(word_id)
            df.at[index, 'label'] = label
        else:
            print(f"[!] 잘못된 파일명 형식: {file_name}")
    except Exception as e:
        print(f"오류 발생: {file_name} - {str(e)}")

# CSV 파일로 저장 (라벨을 마지막 열에 추가)
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"CSV 파일에 라벨 추가 완료: {output_csv}")



# CSV 파일 경로
# input_csv = r"C:\4-2\deeplearning\project\aiHub\data\data_labeled.csv"
# output_csv = r"C:\4-2\deeplearning\project\aiHub\data\data.csv"
#
# # CSV 파일 로드
# df = pd.read_csv(input_csv)
#
# # 라벨 열의 중복 단어 수 계산
# label_counts = Counter(df['label'])
#
# # 150개를 초과하는 단어 목록
# excess_labels = [label for label, count in label_counts.items() if count > 150]
#
# print(f"150개를 초과하는 단어 개수: {len(excess_labels)}")
# print(f"제거할 단어 목록: {excess_labels}")
#
# # 150개를 초과하는 단어를 제거하고, 150개인 단어는 유지
# cleaned_df = df[~df['label'].isin(excess_labels)]
#
# # 제거 후 데이터 수 확인
# print(f"제거 후 데이터 수: {len(cleaned_df)}")
#
# # 필터링된 CSV 파일 저장
# cleaned_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
#
# print(f" 완료: {output_csv}")

