# 범용 카메라·RealSense 도구

이 폴더에는 특정 task의 장면 판정과 무관하게 카메라를 탐색·연결·점검하거나,
RealSense RGB/depth 스트림을 확인하는 도구를 둡니다.

```bash
# recording.env의 TOP_CAM / WRIST_CAM으로 바로 열기
python scripts/piper/camera/realsense_view.py
```

- `camera_check.py`: OpenCV 카메라 인덱스와 기본 영상 스트림을 확인합니다.
- `camera_parallel_connect_test.py`: 두 RealSense 카메라 병렬 연결을 점검합니다.
- `realsense_view.py`: 인자 없이 실행하면 `configs/recording.env`의 `TOP_CAM`/`WRIST_CAM`을
  cycle 화면으로 바로 엽니다. 장치 목록·개별·동시 보기와 depth 컬러맵도 지원합니다.
- `realsense_depth_record_test.py`: depth 기록·양자화·저장 경로를 점검합니다.
- `depth_video_viewer.py`: 저장된 depth 영상을 mm 단위 컬러맵으로 확인합니다.

도형 지우기 보드의 위치·영역·잉크 상태처럼 task 장면 가정이 포함된 도구는
`scripts/tasks/erase_shape/analysis/`에 둡니다. `camera_overlay_view.py`가 그 예입니다.
