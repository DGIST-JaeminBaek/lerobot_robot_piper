# 도형 지우기 화이트보드 증강

Top View 녹화에 목표 도형과 다른 종류의 방해 도형 PNG를 합성해, 도형 지우기 VLA
학습용 영상을 만드는 task 전용 패키지입니다. 로봇에는 명령을 보내지 않습니다.

- `augment_whiteboard.py`: 단일 Top View 영상의 safe 영역에 방해 도형을 합성합니다.
- `batch_augment_dates.py`: episode별 목표 위치를 읽어 단일 합성기를 일괄 실행합니다.
- `analyze_target_positions.py`: 보드 ROI 기준으로 목표 도형이 top/bottom 중 어디에 있는지
  분석합니다.
- `candidate_assets/records_105/`: circle·rectangle·triangle 각각 105개, 총 315개
  검수 완료 RGBA PNG와 원래 crop 위치 인덱스입니다.
- `board_roi.json`: 합성·위치 분석의 공통 화이트보드 좌표계입니다.
- `extract_first_frames.py`, `crop_*_frames.py`: 후보 에셋을 새로 추출할 때만 쓰는 준비 도구입니다.
- `review_asset_quality.py`, `label_asset_candidates.py`, `compare_flagged_assets.py`:
  후보 PNG의 품질과 채택 여부를 검토합니다.
- `build_asset_index.py`, `render_all_assets_with_regions.py`,
  `draw_dual_distribution_regions.py`: 후보의 원래 위치 인덱스·분포를 만듭니다.
- `src/`: 위 합성기의 내부 모듈입니다. 이름은 기존 CLI import 호환성을 위해 유지합니다.
- `review/`: 현재 후보 풀의 최종 판정 CSV·기준 보드 프레임·분포 결과입니다.

`realsense_region_monitor.py`는 같은 코드가 이미 상위 task의
[`../analysis/realsense_region_monitor.py`](../analysis/realsense_region_monitor.py)에 있으므로
여기에는 중복 보관하지 않습니다.

세부 규칙과 에셋 교체 절차는 [docs/README.md](docs/README.md)를 봅니다. 새 산출물은
기본적으로 이 폴더 밖의 `outputs/`에 지정해 보관합니다.
