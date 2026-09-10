# 도형 지우기 데이터셋 도구

이 폴더에는 도형 지우기 task의 원본 녹화를 학습·평가 데이터셋으로 만들기 위한
프레임 범위·라벨·매니페스트 처리 도구를 둡니다.

- `episode_segmentation.py`: 지우기 시연의 시작·종료 이벤트를 찾는 공통 규칙입니다.
- `source_resolver.py`: 매니페스트의 원본 dataset 경로를 안전하게 해석하고 동명이인 후보를 막습니다.
- `autofill_frame_ranges.py`: 매니페스트의 누락된 frame 범위를 자동 제안·채웁니다.
- `export_cut_plan.py`: frame 범위 매니페스트를 JSON·CSV·ffmpeg 컷 계획으로 내보냅니다.
- `prepare_erase_shape_dataset.py`: 선택한 원본 구간을 LeRobot 학습 데이터셋으로 변환합니다.
- `make_done_labels.py`: 잉크 잔량을 바탕으로 done label과 제외 목록을 만듭니다.

이 도구들은 도형·잉크·그리퍼 해제 시점에 대한 task 전용 규칙을 사용합니다.
