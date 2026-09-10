# 테스트

이 폴더는 하드웨어 없이 자동 실행할 수 있는 단위·mock 테스트만 둡니다. 실제 CAN,
카메라, 녹화 데이터셋을 열어 상태를 점검하는 도구는 테스트가 아니라
`scripts/piper/validation/`에 둡니다.

- `piper/`: 어떤 작업에도 재사용하는 Piper 기반 기능의 테스트입니다.
  - `inference/`: 정책 추론, 스무딩, 안전한 리플레이·사전 확인을 검증합니다.
  - `recording/`: 녹화 관측값, 데이터셋 feature, 시작 프레임 보정을 검증합니다.
  - `safety/`: effort 차단, 토크 해제, 명령 EMA를 검증합니다.
- `tasks/erase_shape/`: 도형 지우기 과제의 게이트, HIL, 채점 규칙을 검증합니다.

테스트 대상 구현은 `scripts/piper/` 또는 `scripts/tasks/`에 있으며, 이 폴더는 그
구현을 소비하는 테스트만 둡니다. 재사용 범위에 맞춘 이 분류 경계를 유지합니다.

전체 실행은 프로젝트 루트에서 다음과 같이 합니다.

```bash
python -m pytest scripts/tests
```

일부 테스트는 랩 환경의 패치된 LeRobot 또는 선택 의존성이 필요하며, 해당 환경이
없으면 테스트 자체의 안내에 따라 건너뜁니다.
