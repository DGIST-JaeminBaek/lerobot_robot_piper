# 범용 하드웨어 도구

이 폴더에는 특정 task와 무관하게 Piper의 CAN 연결, 관절 기준, 기구학, 토크 제어를
점검하거나 복구하는 도구를 둡니다. 일부 도구는 실제 로봇을 움직이거나 토크를
인가하므로 실행 전 해당 파일의 주의사항을 확인해야 합니다.

- `setup_can.sh`: 지정한 CAN 인터페이스를 초기화합니다.
- `recover_can_after_reboot.sh`, `detect_can_roles.py`: 재부팅 뒤 CAN 역할을 판별·복구합니다.
- `check_startup_push.py`, `check_zero_reference.py`: 시작 시 목표 밀림과 관절 영점 기준을 점검합니다.
- `joint_check.py`: follower와 선택적 leader의 관절값 수신·all-zero·정규화 범위를 점검합니다.
- `kinematics_check.py`: SDK FK와 펌웨어 EEF 피드백을 비교합니다.
- `piper_mit_probe.py`: 관절별 MIT(임피던스) 제어를 낮은 게인부터 점검합니다.
- `safe_release_torque.py`: 안전한 자세·순서로 arm 및 gripper torque를 해제합니다.
- `align_replay_start.py`: 단일 dataset/episode의 첫 유효 action까지 follower를 저속으로
  정렬합니다. 성공 후 표준 `lerobot-replay`를 별도로 실행하는 사전 단계입니다.

`align_replay_start.py`는 현재 Teleop GUI의 `Replay (Real Robot)`이나 `scripts/6__replay.sh`에
**아직 통합되지 않았습니다.** 필요할 때 수동으로 `--dry-run`으로 시작 자세 차이를 확인하고,
명시적 `--confirm ALIGN_REPLAY_START` 뒤에 실행합니다.

카메라 영상이나 데이터셋 자체의 품질 검증은 `scripts/piper/camera/` 및
`scripts/piper/validation/`에 분리합니다.
