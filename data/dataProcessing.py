import os
import json
import csv

# 사용할 keypoint 종류 (2D만 사용)
KEYPOINT_TYPES = [
    "pose_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "face_keypoints_2d"
]

# 키포인트 접두사
KEYPOINT_PREFIXES = ["pose", "left_hand", "right_hand", "face"]
# 키포인트 길이 (각 유형별로)
KEYPOINT_LENGTHS = [50, 42, 42, 140]  # (x, y) 좌표만 사용하여 길이 변경

# base 경로 설정
keypoint_base_dir = r"C:\4-2\deeplearning\project\aiHub\data\09_real_word_keypoint\keypoint\17"

# 최대 처리 파일 수
MAX_FILES = 30

# CSV 파일 경로
output_csv = r"C:\4-2\deeplearning\project\aiHub\data\data_no_label.csv"

# CSV 파일 저장 준비
with open(output_csv, "w", newline="", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # CSV 헤더 작성 (특징점 + 라벨 제외)
    header = ["file_name"]
    for prefix, length in zip(KEYPOINT_PREFIXES, KEYPOINT_LENGTHS):
        header += [f"{prefix}_f{i}" for i in range(length)]
    writer.writerow(header)

    # 폴더 순회
    for subdir, _, files in os.walk(keypoint_base_dir):
        print(f"[디버그] 현재 폴더: {subdir}")

        # 각 폴더 안의 모든 JSON 파일 탐색 (최대 30개)
        file_count = 0

        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(subdir, file_name)
                clean_file_name = file_name.split('_')[0] + "_" + file_name.split('_')[1] + "_" + file_name.split('_')[2]

                try:
                    # JSON 파일 열기
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            json_data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"[!] JSON 파싱 오류: {file_path}")
                            continue

                    # JSON 로드 실패
                    if json_data is None:
                        print(f"[!] JSON 파일 로드 실패: {file_path}")
                        continue

                    # 데이터 구조 확인
                    if not json_data or "people" not in json_data:
                        print(f"[!] 데이터 없음 또는 구조 오류: {file_path}")
                        continue

                    # 첫 번째 사람의 특징점 가져오기(없을것을 대비)
                    person = json_data["people"]
                    if not person:
                        print(f"[!] 'people' 키가 비어 있음: {file_path}")
                        continue

                    #특징점 추출 (2D 키포인트만 사용, 신뢰도 제거)
                    feature_vector = []
                    for keypoint, prefix, length in zip(KEYPOINT_TYPES, KEYPOINT_PREFIXES, KEYPOINT_LENGTHS):
                        points = person.get(keypoint, [])
                        extracted_points = []
                        for i in range(0, len(points), 3):
                            if i + 1 < len(points):
                                x = points[i] if i < len(points) else 0
                                y = points[i + 1] if i + 1 < len(points) else 0
                            extracted_points.extend([x, y])
                        # 부족한 경우 길이 맞추기
                        if len(extracted_points) < length:
                            extracted_points += [0] * (length - len(extracted_points))
                        feature_vector.extend(extracted_points)

                    # CSV 한 행 작성 (라벨 없이)
                    row = [clean_file_name] + feature_vector
                    writer.writerow(row)

                    # 파일 개수 제한 체크
                    file_count += 1
                    if file_count >= MAX_FILES:
                        print(f"[디버그] 최대 파일 수 {MAX_FILES}개 처리 완료: {subdir}")
                        break

                except Exception as e:
                    print(f"오류 발생: {file_path} - {str(e)}")

print(f"CSV로 저장 완료: {output_csv}")
