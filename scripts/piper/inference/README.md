# 범용 정책 추론 도구

이 폴더는 어떤 Piper task에도 공통으로 적용되는 정책 추론 경로입니다. 특정 도형,
잉크 판정, 특정 데이터셋 이름 같은 task 전용 가정은 포함하지 않습니다.

- `piper_infer_runner.py`: 실시간 추론 제어 루프 본체. GUI와 `teleop_ui`가 함께 사용합니다.
- `action_smoothing.py`: action EMA·temporal ensemble 등 순수 numpy 스무딩 로직입니다.
- `piper_infer_gui.py`: 스무딩·RViz·안전 전송 옵션을 조절하는 실시간 GUI입니다.
- `piper_infer_preview.py`: 데이터셋 observation 전체에서 정책 출력을 추론해 RViz로 미리 확인합니다.
- `piper_human_approved_inference.py`: 사람 승인 단위로 정책 행동을 실행·검토합니다.
- `piper_offline_chunk_rollout.py`, `piper_offline_rollout_rviz.py`: 실제 로봇 없이
  offline rollout과 RViz 재생을 수행합니다.
- `piper_smoothing_sweep.py`: 같은 추론 경로로 스무딩 파라미터를 비교합니다.
- `inference_runtime.py`: 정책 load, dataset/live observation 변환, action chunk 예측을 공유합니다.

RViz용 관절 단위 변환은 상위 `scripts/piper/rviz_joint_state.py`를 공통으로 사용합니다.

실제 task의 종료 게이트, HIL 개입, 성공 판정은 이 폴더가 아니라
`scripts/tasks/<task>/`에 둡니다. task 코드가 이 폴더를 사용할 수는 있지만, 반대
방향 의존성은 만들지 않습니다.
