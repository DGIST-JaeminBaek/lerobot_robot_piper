# 다음 단계: 실물에서 측정/확인해야 할 것

`synthetic/` 아래 코드(6A~6E: 이미지 전처리, board↔base transform solver,
Cartesian trajectory, IK/action 변환, offline preview)는 전부 완성돼 있고
테스트도 통과한다. 하지만 지금까지 쓰인 board 크기, 기준점, 높이, 자세 값은
전부 가상/예시(illustrative)이지 실측이 아니다. 아래 항목을 실물에서 확인해야
`synthetic/configs/*.example.json`을 실제 값으로 바꿔서 진짜 trajectory를
생성할 수 있다.

## 순서

### 1. 물리적 고정

- [ ] 카메라, 화이트보드, 로봇 base를 전부 고정한다. 이후 하나라도 움직이면
      아래 모든 측정이 무효가 된다.

### 2. 지우개 거치대 + 깔끔한 GRASP/PLACE 시연 1개

지우개를 매번 정확히 같은 방식으로 쥐어야 그리퍼→지우개 끝 사이의 물리적
오프셋(TCP offset)이 항상 동일하게 취급된다. 이건 손으로 매번 신경 쓸 게
아니라 구조적으로 보장하는 게 낫다.

- [ ] 지우개를 놓을 **고정 거치대**를 만들고 위치를 고정한다.
- [ ] teleop으로 거치대에서 지우개를 집는 **깔끔한 GRASP 궤적 하나**를
      녹화한다 (필요하면 노이즈에 안 깨지도록 여러 개).
- [ ] 지우개를 다시 거치대에 놓는 **PLACE 궤적**도 녹화한다 — 매번 같은
      자세로 안착해야 다음 GRASP도 같은 그립이 된다.
- [ ] 거치대가 촬영/측정 도중 밀리지 않는지 중간중간 확인한다.

이 두 궤적(GRASP, PLACE)이 `synthetic/trajectory/compose.py`의
`require_recorded_template()`이 요구하는 실제 template가 된다.

### 3. 지우개 쥔 채로 도달 범위 훑기 (안전 확인용)

- [ ] 2번에서 지우개를 집은 상태로 팔을 이리저리 움직여 보며, 팔 자신이나
      케이블·테이블과 부딪히지 않는 안전한 범위를 감으로 확인한다.
- [ ] 이 범위 안에 보드 작업 영역 전체가 들어오는지 확인한다.

이 단계는 board_xy↔base_xyz 대응점을 만들어주지는 않는다 — 순수히 안전
범위 확인용이다.

### 4. 보드 실측

- [ ] 화이트보드의 실제 가로·세로(mm)를 잰다.
- [ ] 로봇이 안전하게 닿을 수 있는 내부 작업 영역(직사각형)을 정한다 —
      보드 전체가 아니라 3번에서 확인한 안전 범위 안쪽으로.

### 5. TOP 이미지 기준점 선택 (도구 이미 있음)

```bash
python synthetic/calibration/select_board_points_web.py \
  --video <TOP mp4> --frame <episode 시작 frame> \
  --unit mm --board-width <4번 가로> --board-height <4번 세로> \
  --output synthetic/outputs/board_points_epNNN.json
```

- [ ] 5~10개 episode에서 반복 (`README.md` "여러 episode의 선택 결과 집계"
      참고, `aggregate_board_points.py`로 집계)
- [ ] `solve_homography.py` / `validate_calibration.py`로 image↔board
      검증 (reprojection error 확인)

### 6. Board 기준점의 실제 EEF 좌표 측정 (지우개를 쥔 상태로)

- [ ] 2번에서 녹화한 GRASP를 재생해 지우개를 집는다.
- [ ] 5번에서 정한 board 기준점(3개 이상, 일직선 아니게) 각각에 **지우개
      끝**이 닿도록 teleop한다.
- [ ] 그 순간 `CalFK()` 또는 `GetArmEndPoseMsgs()`로 EEF pose를 기록한다
      (mm, degree).
- [ ] 이때 지우개가 보드를 누르는 **자세(orientation)** 를 실제로 지우기 때
      쓸 고정 자세 그대로 유지한다 — 이 자세가 7번의 `tool_rpy_deg`가 된다.

이렇게 측정하면 TCP offset을 별도 숫자로 뺄 필요가 없다 — 지우개 끝이 실제로
그 board 점에 닿았을 때의 EEF 값이므로 오프셋이 이미 포함돼 있다. 단, 매번
2번의 같은 GRASP로 잡은 상태에서 재야 한다.

- [ ] 이 대응점들을 `synthetic/configs/board_base.example.json` 형식으로
      정리해서 `synthetic/transforms/solve_board_base.py --correspondences`로
      board↔base transform을 계산한다.

### 7. 접촉/비접촉 높이와 고정 tool 자세 확정

- [ ] 6번에서 지우개가 보드에 닿을 때의 높이를 **contact_height_mm**으로,
      팔에 무리 안 가면서 안전하게 이동하는 높이를 **hover_height_mm**으로
      정한다 (녹화 데이터에서 역산하거나 직접 재는 방법 둘 다 가능).
- [ ] 6번에서 쓴 접촉 자세를 **tool_rpy_deg**로 확정한다.
- [ ] `synthetic/trajectory/compose.py`의 `BoardMotionConfig`에 이 값들을
      채운 JSON을 만든다.

### 8. Effort 분포 측정 / safety threshold 재검증

- [ ] 정상 자유 이동과 실제 접촉 시의 effort 값을 모아서 분포를 본다.
- [ ] 현재 `SAFETY_EFFORT_LIMIT=8.0`이 이 작업(지우기 접촉)에 적절한지
      확인하고, 아니면 새 값을 정한다. 지금 이 값은 이 작업용으로 검증된
      값이 아니다.

### 9. IK 초기 seed 확인

- [ ] trajectory 생성 시작점으로 쓸 실제 관절 자세(예: parking 또는
      pregrasp 부근)의 joint 값을 확인해 `--initial-seed-joint-rad`로 쓴다.

## 값이 갖춰지면

위 값들을 주시면 아래를 실제 값으로 재실행해서 진짜(가상 아님) offline
trajectory를 생성할 수 있다:

- `synthetic/transforms/solve_board_base.py` — 6번 대응점으로 실제
  board↔base transform
- `synthetic/preview/generate_preview.py` — 7, 9번 값 + 5/6번 결과로
  실제 preview/validation report

## 아직 범위 밖 (다음 인계)

- 6단계: 사람 승인형 실물 실행 adapter (`offline`/`rviz-only`/`real` 모드,
  단계별 승인, `PiperFollower.send_action()` 연결) — 이번 인계에는 포함되지
  않았다.
- 위 1~9번이 갖춰지고 offline 검증을 통과한 뒤에만 시작한다.
