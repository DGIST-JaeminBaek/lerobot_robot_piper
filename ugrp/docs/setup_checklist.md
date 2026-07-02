# 실험 전 체크리스트

## 소프트웨어

- Python 가상환경 활성화
- `lerobot_robot_piper` 설치 완료: `pip install -e .`
- `lerobot-record`, `lerobot-teleoperate` 명령 실행 가능
- `piper_sdk`, `wego_piper` import 가능

## 하드웨어

- leader Piper 연결
- follower Piper 연결
- CAN-USB 어댑터 연결
- 상방 카메라 연결
- 팔목 카메라 연결
- 비상 정지 또는 전원 차단 준비

## CAN

```bash
bash scripts/setup_can.sh
```

`can_leader1`, `can_follower1`처럼 고정 이름이 필요하면 `piper-setup` 사용.

서버에서 `sudo` 비밀번호 입력이 필요한 경우, 녹화 전에 터미널에서 CAN 초기화 먼저 실행.

## 카메라

```bash
python /path/to/lerobot_robot_piper/camera_check.py
```

- 상방 카메라 video 번호 확인
- 팔목 카메라 video 번호 확인
- 확인한 번호를 `configs/recording.env`의 `TOP_CAM`, `WRIST_CAM`에 입력

## 캘리브레이션

두 팔이 같은 물리 자세에서 다른 값을 보이면 녹화 전 `piper-calibrate` 실행.

초기 자세가 불확실하면 `MAX_RELATIVE_TARGET`을 낮춰서 텔레옵 점검.

```bash
MAX_RELATIVE_TARGET=2 bash scripts/teleop_smoke_test.sh
```
