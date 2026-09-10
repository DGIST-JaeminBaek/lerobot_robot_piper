# scripts 안내

이 폴더는 Piper 로봇 실험에 쓰는 실행 진입점과 보조 도구를 **재사용 범위**로 나눈다.

## 빠른 실행

번호형 셸 스크립트는 일반적인 실험 순서를 제공한다.

- `0__launch_gui.sh`: CAN 사전 점검 후 Teleop UI 실행. 시스템 Tk/Noto 한글 렌더링도 여기서 적용한다.
- `1__init_can.sh`: CAN 인터페이스 초기화·이름 배정
- `2__find_camera.sh`: 카메라 탐색
- `4__teleoperate.sh`: leader/follower teleoperation
- `5__record.sh`: LeRobotDataset 녹화
- `6__replay.sh`: 데이터셋 재생
- `7__train.sh`: 정책 학습
- `10__qc_studio.sh`: 도형 지우기 QC Studio
- `13__erase_gate.sh`, `13__eval_session.sh`: 도형 지우기 실행·평가

실험 전 설정은 `configs/recording.env`에서 관리한다. 전체 실행 절차는
[`docs/operations.md`](../docs/operations.md)를 참고한다.

## 폴더 구조

```text
scripts/
├── piper/              # task와 무관하게 재사용하는 Piper 도구
│   ├── inference/      # 정책 추론, smoothing, RViz 사전 확인
│   ├── recording/      # 단일 녹화, 시작 프레임·action 보정
│   ├── validation/     # replay, 세션 점검, 데이터셋 검증
│   ├── hardware/       # CAN, 기구학, 토크·MIT 안전 점검
│   └── camera/         # OpenCV·RealSense 카메라 점검
├── tasks/
│   └── erase_shape/    # 도형 지우기 task 전용 도구
│       ├── runtime/    # HIL 실행·클러치·상태 표시
│       ├── evaluation/ # 잉크 판정, 평가 UI, eval_kit
│       ├── dataset/    # 라벨·구간·학습 데이터셋 생성
│       ├── qc/         # 프레임 범위 QC와 검토 도구
│       ├── analysis/   # 카메라·자세·퇴각 분석
│       └── lib/        # task 내부 공용 Python 보조 모듈
├── tests/              # 하드웨어 없이 실행하는 단위·mock 테스트
├── training/           # GPU별 학습 실행 프리셋
├── archive/            # 이전 구현 보관
└── lib/                # 번호형 셸 런처 공통 함수
```

의존성 방향은 `tasks/* → piper/*`만 허용한다. 범용 `piper/` 도구는 특정 task의
장면·성공 기준·데이터셋 이름을 알지 않아야 한다.

## 테스트

프로젝트 루트에서 `ugrp` 환경을 활성화한 뒤 실행한다.

```bash
python -m pytest scripts/tests -q
```
