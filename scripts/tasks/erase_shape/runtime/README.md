# 도형 지우기 실행 도구

이 폴더는 "지우개를 집어 도형을 지운다" task를 실제로 실행하는 제어·상태 도구입니다.
정책 추론 자체는 범용 `scripts/piper/inference/`를 사용하고, 여기서는 task 전용
종료 게이트와 사람 개입(HIL)을 더합니다.

- `erase_run.py`: 정책 실행 → park → 잉크 판정 → 필요 시 재시도하는 closed-loop 러너입니다.
- `hil_clutch.py`: 리더암을 통한 HIL 개입의 키보드 토글·델타 인계 규칙입니다.
- `erase_hil_panel.py`: HIL 실행 중 상태와 제어권을 표시하는 패널입니다.
- `erase_status.py`: 실행 상태·측정값을 터미널에 일관되게 표시합니다.

실물 명령이 가능한 도구이므로 `erase_run.py`의 확인 옵션과 `13__erase_gate.sh`의
`STAGE=dry` 절차를 먼저 확인해야 합니다.
