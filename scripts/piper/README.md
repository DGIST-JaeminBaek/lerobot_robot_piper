# 범용 Piper 도구

이 폴더에는 특정 작업의 장면·성공 기준·데이터셋 이름에 의존하지 않고, Piper 로봇의
여러 작업에서 재사용하는 도구를 둡니다.

- `inference/`: 정책 추론, action smoothing, 사전 검증, RViz 시각화
- `recording/`: 단일 episode 녹화와 action·시작 프레임 보정
- `validation/`: 세션 준비, replay, 데이터셋·영상 검증
- `hardware/`: CAN, 기구학, 토크·MIT 안전 점검
- `camera/`: OpenCV·RealSense 카메라 탐색·연결 점검
- `rviz_joint_state.py`: Piper 정규화 action을 RViz `JointState` 물리 단위로 바꾸는 공용 변환입니다.

도형 지우기처럼 작업 전용 목표·판정 기준을 포함한 도구는 `scripts/tasks/`에 둡니다.
