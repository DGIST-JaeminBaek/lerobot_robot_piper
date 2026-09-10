# 범용 추론 테스트

정책 모델의 action chunk를 Piper 명령으로 연결하는 공통 경로를 검증합니다.

- `test_action_smoothing.py`: numpy 기반 action smoothing 수학
- `test_infer_runner_mock.py`: 실시간 추론 러너의 모드·기록 처리
- `test_human_approved_inference_mock.py`: 사람 승인 전에는 로봇 명령이 나가지 않는지

실제 정책, 카메라, CAN에 연결하지 않고 mock 또는 합성 입력만 사용합니다.
