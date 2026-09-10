# Assets

candidate_assets/records_105/는 현재 후보 에셋 315개의 canonical 경로다.

- shape 폴더: RGBA PNG
- manifest CSV: 원본 프레임과 crop bounds
- asset_index.csv: 배치용 metadata index. `source_center_x/y`는 도형 bbox 중심, `source_x/y`는 실제 PNG crop의 원래 top-left 좌표

수동 교체 후에는 manifest bounds와 asset_index.csv를 함께 갱신한다.

## Directory policy

- candidate_assets/records_105/: 현재 사용할 후보만 보관
- source_frames/: 재추출과 원본 비교를 위한 프레임 보관
- review/: decision CSV, 기준 보드 프레임, 최종 분포 이미지 보관

## Replacement checklist

1. 새 PNG가 전체 선과 적절한 여백을 포함하는지 확인한다.
2. 기존 후보를 교체한다.
3. shape manifest의 bbox와 crop bounds를 수정한다.
4. build_asset_index.py를 다시 실행한다.
5. 합성 검수로 원래 위치에서 자연스러운지 확인한다.
