# Pipeline

1. 첫 프레임에서 보드 ROI와 목표 도형 위치를 분석한다.
2. 로봇, 케이블, 기존 그림, 움직임, 보드 외곽을 unsafe로 제외한다.
3. 목표 도형과 반대쪽 보드 반쪽에 방해 도형을 배치한다.
4. 목표 도형과 같은 종류의 방해 도형은 사용하지 않는다.
5. safe 위치를 먼저 계산한 뒤 원래 추출 위치가 가까운 에셋을 선택한다.

asset-index 옵션이 위치 호환 선택을 활성화한다. 배치마다 아직 사용하지 않은 에셋 중 원래 crop 좌표가 safe 영역에 그대로 들어가는 에셋을 먼저 비복원으로 고른다. 그런 에셋이 없을 때만 원래 위치에 가장 가까운 safe 좌표로 fallback한다.

## Inputs

- input video: Top View MP4
- board_roi.json: 보드 좌표계
- candidate_assets/records_105/: shape별 RGBA PNG
- asset_index.csv: 각 PNG의 원본 crop 위치와 크기

## Outputs

- augmented.mp4: 합성 영상
- placement_metadata.json: 선택한 에셋과 최종 배치 좌표
- debug/: 옵션 사용 시 safe mask와 preview

## Compatibility mode

asset-index를 생략하면 에셋을 먼저 고른 뒤 safe 위치를 찾는다. 기본 에셋 루트는 후보 315개 세트이며, 원래 위치 우선 정책을 쓰려면 assets-root와 asset-index를 함께 지정한다.
