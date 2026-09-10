# 범용 Piper 테스트

특정 작업의 성공 기준이나 장면 가정 없이, Piper 로봇에서 재사용되는 기능을
검증합니다. 이 폴더의 테스트 대상 구현은 `scripts/piper/`에 둡니다.

- `inference/`: 모델 출력 처리, 스무딩, 승인형 실행, 안전한 리플레이
- `recording/`: 관측 feature와 녹화 데이터셋 형식
- `safety/`: 외력 안전 차단과 torque 해제
- `validation/`: dataset replay의 일반/RViz CLI 경계와 Piper 관절 변환

도형의 잉크 잔량, 목표 도형, HIL 평가처럼 과제 정의가 필요한 테스트는 여기 두지
않고 `../tasks/erase_shape/`에 둡니다.
