# Tools

## Review

- review_asset_quality.py: 컨택트시트와 alpha report
- label_asset_candidates.py: Good, Bad, Hold 합성 검수
- compare_flagged_assets.py: flagged 에셋의 원본 비교

## Distribution

- render_all_assets_with_regions.py: 원래 위치 overlay
- draw_dual_distribution_regions.py: 95%·dense 영역

## RealSense

realsense_region_monitor.py는 Top View serial 327122074262와 1280x720 고정 영역을 사용한다. B 키로 clean baseline을 캡처한다. Wrist View는 이 좌표계에 사용하지 않는다. GUI가 없는 SSH 환경에서는 `--snapshot review/realsense_snapshot.png`로 영역 표시가 포함된 한 프레임을 저장해 확인한다.

## What to keep

- 최종 판정: review/candidate_asset_decisions.csv
- 기준 보드: review/whiteboard_last_frame.png
- 최종 공간 분석: review/asset_distributions/all_assets_overlaid_95pct_and_dense_regions.png

다른 review 이미지와 crop debug 이미지는 일회성 검수 산출물이며, 최종 검수 후 보관하지 않는다.
