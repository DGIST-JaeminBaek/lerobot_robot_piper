# 도형 지우기 과제 문서

지우개를 집어 화이트보드 도형을 지우는 `erase the shape` 과제 전용 문서입니다.
공용 Piper 운용 절차는 `../../../operations.md`, 공용 policy 실행 방식은
`../../../policy/README.md`를 참고합니다.

| 범주 | 문서 | 내용 |
| --- | --- | --- |
| 평가 | [evaluation/eval_session_howto.md](evaluation/eval_session_howto.md) | 판정 방법 + 실물 평가 세션 실행 절차 |
| 평가 | [evaluation/reach_session_howto.md](evaluation/reach_session_howto.md) | 도형 선택(도달)만 보는 축소판 세션 — 영상 없음, 사람 판정 |
| 품질 관리 | [quality/qc_and_trimming.md](quality/qc_and_trimming.md) | 녹화 QC와 frame 구간 자르기 |
| 런타임 | [runtime/erase_run_design.md](runtime/erase_run_design.md) | 잉크 판정, 종료 설계 (지금 쓰는 평가 방식) |
| 런타임 | [runtime/hil_intervention_design.md](runtime/hil_intervention_design.md) | HIL(사람 개입) 설계 — 보류, 지금 안 씀 |
| 학습 | [training/method.md](training/method.md) | 기본 학습 파이프라인 |
| 학습 | [training/training_status.md](training/training_status.md) | 현재 학습 캠페인 현황 |
| 보고서 | [reports/](reports/) | 비교·재보정 평가 결과 |

`scripts/tasks/erase_shape/`는 구현 코드, 이 폴더는 그 코드의 사용법·근거·실험
기록을 담당합니다.
