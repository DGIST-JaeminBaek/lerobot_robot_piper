# 도형 지우기 장면 분석 도구

이 폴더에는 학습 데이터와 실제 지우기 보드 장면이 일치하는지 확인하는 도구를 둡니다.
따라서 일반 카메라 진단과 달리 보드·도형·블록 위치라는 task 전용 기준을 포함합니다.

- `camera_overlay_view.py`: 지우기 보드 기준 이미지에서 쓰는 영역 좌표를 영상 위에
  겹쳐 보여, camera crop과 작업 영역을 점검합니다.
- `block_alignment_tool.py`: 지우개 블록이 학습 데이터의 기준 위치에 놓였는지
  실시간으로 안내합니다. 로봇 팔에는 명령을 보내지 않습니다.
- `camera_overlay_view_315.py`: 315개 도형 지우기 녹화에서 정한 여섯 대표 위치를
  바로 표시하는 전용 뷰어입니다.
- `realsense_region_monitor.py`: 해당 보드 영역을 기준으로 새 도형이 그려졌는지
  RealSense 컬러 스트림에서 감시합니다.
- `check_exposure_drift.py`: 실행 중 판정 카메라 노출이 기준 프레임에서 얼마나
  달라지는지 측정합니다.
- `shape_position_analysis.py`: 학습 녹화에서 도형이 보드의 어느 위치에 놓였는지
  분석합니다.
- `wrist_ink_probe.py`: 손목 카메라가 잉크 잔량 판정에 쓸 수 있는지 검증합니다.
- `retreat_analysis.py`: 팔이 물러난 뒤에만 판정할 수 있는지 시연 기록으로 분석합니다.
- `first_chunk_fk_analysis.py`: 도형별 episode 첫 action chunk를 Piper FK/EEF 궤적으로
  비교합니다. circle·triangle·rectangle 그룹과 task 전용 frame-range 매니페스트를 사용합니다.
- `joint_state_timeseries.py`: 315개 원본 녹화의 `observation.state` joint1~6·gripper를
  관절별 시계열로 그립니다. EEF로 합쳐 보면 가려지는 축별 움직임과 세션 편차를 봅니다.
- `policy_conditioning_probe.py`: 정책이 도형 위치 같은 **조건부 정보를 실제로 쓰는지**
  측정합니다. 진행률 지점마다 action chunk를 받아 npz로 남깁니다. 로봇을 쓰지 않습니다.
  정책마다 conda 환경이 다르므로(pi0 / ugrp) 따로 실행합니다.
- `compare_policy_conditioning.py`: 위 npz들을 비교해 "예측의 에피소드 간 분산 /
  정답의 분산"을 관절·진행률별로 그립니다. 비율이 0에 가까우면 관측을 무시하고
  평균 궤적을 재생한다는 뜻입니다.

