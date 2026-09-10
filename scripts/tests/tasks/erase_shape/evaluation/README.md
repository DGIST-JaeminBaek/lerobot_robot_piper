# 도형 지우기 실행·평가 테스트

도형 지우기 task의 실행 종료와 성공 판정을 하드웨어 없이 검증합니다.

- `test_erase_run_mock.py`: 시도 단위 게이트와 추론 실행 흐름
- `test_hil_clutch_mock.py`: 리더암 개입(HIL) 클러치 전환과 인계 규칙
- `test_erase_eval_ui.py`: 사람 확인을 포함한 평가 세션 UI의 상태 전이
- `test_qc_metrics_mock.py`: 녹화 QC에서 사용하는 관절 지표
- `test_eval_kit.py`: 합성 보드 이미지로 자동 채점·통계·보정 계산

실제 보드·카메라에서의 판정값 확인은 테스트가 아니라 task 운영 도구에서 수행합니다.
