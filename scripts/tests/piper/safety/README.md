# 범용 안전 테스트

실제 로봇에 명령을 보내기 전 지켜야 할 Piper 공통 안전 규칙을 mock으로 검증합니다.

- `test_action_ema_mock.py`: 급격한 목표 변화 완화와 확인 문구 상수
- `test_release_mock.py`: 안전한 torque 해제와 그리퍼 해제 순서
- `test_safety_mock.py`: effort 과부하 시 hold 또는 park로 전환하는지
