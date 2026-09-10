# 범용 검증 도구

이 폴더에는 특정 task의 성공 기준과 무관하게 Piper의 세션 준비, 녹화 재생,
데이터셋 구조·영상 품질을 확인하는 도구를 둡니다.

- `piper_replay_player.py`: episode 영상·관절값을 재생합니다. 창 하단 `Frame` 슬라이더로
  원하는 프레임으로 이동할 수 있고, `--rviz`를 붙이면 같은 프레임의 Piper action을
  ROS2 `JointState`로 publish해 RViz와 동기화합니다.
- `batch_replay_review/`: 여러 녹화를 순회하며 눈검사하고, 명시적 확인 뒤 실물 재생도 지원합니다.
- `check_effort.py`, `effort_audit.py`: 한 데이터셋 또는 여러 기록의 effort 분포를 점검합니다.
- `dataset_quick_check.py`: all-zero state·action 범위·급변·frame 연속성을 빠르게 확인합니다.
  인자 없이 실행하면 `configs/recording.env`의 `DATASET_ROOT`를 사용합니다.
- `dataset_structure_check.py`: metadata와 action/state feature 이름, 지정 episode의 기본 구조를 확인합니다.
- `decode_check_parallel.py`: 학습 전에 전 영상 프레임을 실제로 병렬 디코딩해 손상 여부를 확인합니다.
- `preview_video_crop.py`: 학습 전 TOP/WRIST 영상 crop을 비파괴적으로 미리 봅니다.

특정 작업의 성공 판정·장면 기준·전용 QC는 이 폴더에 두지 않습니다.

실물 replay 전에 현재 follower를 첫 action에 맞추려면
`scripts/piper/hardware/align_replay_start.py`를 수동으로 실행할 수 있습니다. 현재 Teleop GUI의
`Replay (Real Robot)`, `scripts/6__replay.sh`, batch replay는 이 정렬 단계를 자동 호출하지 않습니다.
