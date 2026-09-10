# 범용 녹화 테스트

Piper 관측값이 녹화 데이터셋으로 안전하게 저장·보정되는 공통 규칙을 검증합니다.

- `test_effort_mock.py`: effort·velocity 관측값의 on/off 분기
- `test_dataset_features_mock.py`: 관측 feature가 LeRobot 데이터셋 schema에 들어가는 방식
- `test_smooth_start_mock.py`: 시작 프레임 보정 시 위치만 수정하고 effort·velocity는 보존하는지
- `test_fix_action_offset.py`: action offset 보정의 순수 계산 로직
