# 실물 평가 세션 실행 설명서

지우기 평가 GUI(`erase_eval_ui.py`)로 실물 롤아웃을 돌리는 **실행 절차** 문서.
채점 지표·통계 근거는 [평가 프로토콜](evaluation_protocol.md)을 볼 것 — 여기는
"환경을 어떻게 세팅하고 어떤 명령을 치는지"만 다룬다.

작성 2026-08-18. 이 세션에서 실물로 검증된 절차와 실제로 겪은 문제·수정 사항을
그대로 반영했다.

---

## 1. 환경 준비 (매번)

```bash
conda activate ugrp
source /opt/ros/humble/setup.bash
source ~/UGRP/ros2_ws/install/setup.bash
cd /home/ugrp43/UGRP/lerobot_robot_piper
```

**`ugrp` 환경을 쓸 것.** `conda activate pi0`는 pi0 체크포인트를 돌릴 때만
쓴다(§5 참고). 잘못된 환경(예: base python)으로 채점 스크립트를 돌리면
`pyarrow`가 없어서 그리퍼 로그를 못 읽고, 종료 사유가 전부 `unknown(no_log)`이
되면서 아무 경고 없이 조용히 망가진다 — 반드시 `conda activate ugrp` 확인 후
시작할 것.

카메라 시리얼(`configs/recording.env` 기준, 카메라를 옮기지 않았다면 고정값):

| 카메라 | 시리얼 |
|---|---|
| top | `327122074262` |
| wrist | `243322071626` |

---

## 2. 보드 상태 확인 (매 세션 첫 순서)

카메라가 조금이라도 움직이면 보드 좌표가 틀어진다. 실행 전에 눈으로 확인:

```bash
python scripts/tools/ink_metric.py <아무_기존_에피소드_폴더> --dump-roi /tmp/roi.png
```

`/tmp/roi.png`에서 볼 것:

- 보드 영역(파란 사각형)이 실제 화이트보드와 맞는지
- 그려진 도형에 bbox(빨간 사각형)가 정확히 씌워지는지
- **로봇 팔·나무 지우개 블록이 도형으로 잘못 잡히지 않는지**
- 노란 사각형(테이프 자국 배제 구역, `DEFAULT_EXCLUDE`)이 실제 테이프 자국
  위치와 맞는지 — 카메라를 옮겼거나 이물질을 실제로 뗐으면 §7 참고

안 맞으면 `--board X,Y,W,H`로 덮어써서 맞는 값을 찾은 뒤, 그 값을 아래 §4
명령의 `EVAL_BOARD`에 넣는다(생략하면 기본값 `228 0 515 720` 사용).

---

## 3. 체크포인트 선택

`outputs/train/<task>/<run_name>/checkpoints/<step 또는 last>/pretrained_model`
형태다. `last`가 보통 최종 스텝과 같다.

현재 있는 체크포인트(2026-08-18 기준, `ls outputs/train/*/*/`로 최신 목록 재확인할 것):

| 체크포인트 경로 | 종류 | wrist 카메라 | 학습 데이터셋(`--dataset_root`) |
|---|---|---|---|
| `pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_315` | SmolVLA | 있음 | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_toponly_0802_0804_0805_0813pm_135` | SmolVLA | **없음(top-only)** | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0802_0804_0805_0813pm_135_top_only` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_topwrist_0802_0804_0805_0813pm_135` | SmolVLA | 있음 | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0802_0804_0805_0813pm_135` |
| `pick_up_the_eraser_and_erase_the_shape/pi0_lora_pickup_prompt_315` | **pi0 LoRA** | 있음 | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315` |
| `erase_the_shape/smolvla_topwrist_315_r1` | SmolVLA (erase만, pickup 없음) | 있음 | `records/outputs/erase_the_shape_315` |

**`--dataset-root`는 반드시 그 체크포인트를 학습시킨 데이터셋을 가리켜야 한다** —
정규화 통계와 관찰 형태(top-only인지 top+wrist인지)를 여기서 읽는다. 표의 경로가
안 맞으면 아래로 확인:

```bash
cat outputs/train/<task>/<run_name>/checkpoints/last/pretrained_model/train_config.json \
  | python -c "import json,sys; d=json.load(sys.stdin); print(d['dataset']['root'])"
```

**wrist 카메라 유무 확인**:

```bash
cat outputs/train/<task>/<run_name>/checkpoints/last/pretrained_model/config.json \
  | python -c "import json,sys; print(list(json.load(sys.stdin)['input_features']))"
```

`observation.images.wrist`가 없으면 top-only 체크포인트다 — §4 명령에서
`--wrist-crop`을 **비워야** 한다(`--wrist-crop ""`).

---

## 4. GUI로 세션 실행

```bash
EVAL_CUTOFF=90 \
EVAL_MODEL=<모델이름> \
EVAL_CONDITION=<조건이름> \
EVAL_TARGET=<circle|triangle|rectangle> \
EVAL_TRIALS=<목표 시행 수> \
EVAL_WATCH_DIR=records/rollout \
EVAL_ROLLOUT_CMD="python scripts/tools/erase_run.py \
  --policy_path <체크포인트>/checkpoints/last/pretrained_model \
  --dataset_root <학습에 쓴 데이터셋 루트> \
  --target <circle|triangle|rectangle> \
  --top-cam 327122074262 --wrist-cam 243322071626 \
  --top-crop 280,0,720 --wrist-crop 280,0,720 \
  --mode augment --max-attempts 1 --stop-on-release \
  --confirm" \
bash scripts/13__eval_session.sh
```

### 인자 설명

| 인자 | 의미 |
|---|---|
| `EVAL_MODEL` / `EVAL_CONDITION` | 결과 분류용 라벨. 자유 문자열(예: `smolvla`/`pickup315`). 결과 폴더 이름(`evaluation/<MMDD>_<model>_<condition>/`)에 들어간다 |
| `EVAL_TARGET` | 지금 보드에 그려진 도형과 일치시킬 것 |
| `EVAL_TRIALS` | 화면 표시용 목표 횟수일 뿐, 실제 반복 제어는 안 한다 — 원하는 만큼 space로 반복하면 된다 |
| `EVAL_WATCH_DIR` | **필수.** `erase_run.py --mode augment`가 새 에피소드를 만드는 상위 폴더. 항상 `records/rollout` |
| `EVAL_CUTOFF` | 초 단위 시간 제한. 기본 60인데 모델 로딩+940스텝 추론+파킹까지 합치면 60초를 넘기기 쉬우므로 **90 이상 권장**. 파킹이 끝난 뒤(`[DISCONNECT]` 로그 이후)엔 자동으로 무시되므로 너무 타이트하게 잡을 필요는 없다 |
| `--policy_path` | §3에서 고른 체크포인트의 `pretrained_model` 폴더 |
| `--dataset_root` | 그 체크포인트를 학습시킨 데이터셋(§3 표) |
| `--top-crop` / `--wrist-crop` | 전 데이터셋이 `280,0,720`으로 통일돼 있다. **top-only 체크포인트면 `--wrist-crop ""`으로 비울 것** |
| `--mode augment` | 롤아웃을 LeRobotDataset으로 기록(크롭 전 원본 프레임 저장). 기록이 필요 없으면 `--mode demo` |
| `--stop-on-release` | 지우개를 확실히 놓거나(그리퍼가 파지 수준에서 뚜렷이 벌어짐) 관절이 3초 이상 거의 안 움직이면 즉시 파킹으로 간다. **항상 켜둘 것** — 없으면 정책이 다 끝내고도 남은 스텝(최대 940)을 다 채운다 |
| `--confirm` | 실제로 팔을 움직인다. 빼면 인자 검증만 하고 종료(로봇 무동작) |

### 창 조작

| 키 | 동작 |
|---|---|
| space | 시작 / 중단 |
| Enter | 확인하고 다음 시행 |
| n | 이번 시도 버리기 (채점 안 됨) |
| 1~7 | 자동 실패 유형이 틀렸으면 사람이 수정 |
| x | 이 에피소드를 집계에서 제외 |

⚠️ **[중단]은 안전장치가 아니다.** 프로세스를 종료할 뿐이라 팔이 즉시 서지
않을 수 있다. 위험하면 하드웨어 비상정지를 먼저 누르고, 그 다음 이 버튼으로
세션을 정리한다.

### 결과 확인

`evaluation/<MMDD>_<EVAL_MODEL>_<EVAL_CONDITION>/`에 쌓인다.

| 파일 | 내용 |
|---|---|
| `episodes.csv` | 에피소드 1행씩. 여러 번 돌려도 이어붙는다 |
| `summary.md` | 조건별 성공률(95% CI)·진행률·지움 비율·t@90·회복률·실패 유형 분포 표. 보고서에 그대로 붙일 수 있다 |
| `summary.json` | 같은 내용, 스크립트/AI 입력용 |

---

## 5. 모델을 바꿔가며 여러 조건 비교하기

**같은 창을 계속 쓰지 말고, 조건마다 새로 실행한다.** `EVAL_MODEL`/`EVAL_CONDITION`이
결과 폴더를 가르기 때문에, 창을 껐다 다시 §4 명령을 다른 값으로 바꿔 실행하면
된다.

```bash
# 조건 1: SmolVLA top+wrist
EVAL_MODEL=smolvla EVAL_CONDITION=topwrist ... EVAL_ROLLOUT_CMD="... --policy_path .../smolvla_topwrist_.../pretrained_model --dataset_root .../pick_up_the_eraser_..." bash scripts/13__eval_session.sh

# 조건 2: SmolVLA top-only  (주의: --wrist-cam도 빼거나 무시되지만 --wrist-crop은 반드시 비운다)
EVAL_MODEL=smolvla EVAL_CONDITION=toponly ... EVAL_ROLLOUT_CMD="... --policy_path .../smolvla_toponly_.../pretrained_model --dataset_root .../..._top_only --wrist-crop \"\"" bash scripts/13__eval_session.sh
```

**pi0 체크포인트는 환경이 다르다** — `conda activate pi0`로 바꿔야 한다
(transformers 브랜치가 다름, [pi0_finetuning.md](../training/pi0_finetuning.md) 참고).
`ugrp`에 pi0용 transformers를 설치하면 SmolVLA·HAMLET이 깨지므로 **섞지 말 것**.
pi0로 돌릴 땐 새 터미널을 열어 처음부터:

```bash
conda activate pi0
source /opt/ros/humble/setup.bash
source ~/UGRP/ros2_ws/install/setup.bash
cd /home/ugrp43/UGRP/lerobot_robot_piper
# 이후 §4와 동일하게, --policy_path만 pi0 체크포인트로
```

같은 조건(같은 도형, 같은 순서 원칙이면 더 좋음)에서 모델만 바꿔 비교해야
결과가 공정하다 — evaluation_protocol.md §6 "페어링" 참고.

---

## 6. 채점 도구를 GUI 없이 직접 쓰기

이미 녹화된 롤아웃 폴더를 다시 채점하거나, 여러 개를 한 번에 채점할 때:

```bash
python scripts/tools/erase_eval.py 'records/rollout/<데이터셋명>_rollout_*' \
  --model smolvla --condition async --out-dir evaluation/재채점
```

기존 CSV의 요약만 다시 만들 때:

```bash
python scripts/tools/erase_eval.py --summarize evaluation/<폴더>/episodes.csv
```

---

## 7. 알려진 함정

| 함정 | 증상 | 대응 |
|---|---|---|
| `conda activate ugrp` 안 함 | 종료 사유가 전부 `unknown(no_log)` | 환경 확인 후 재실행 |
| top-only 체크포인트에 `--wrist-crop` 그대로 둠 | 관측 형태 불일치로 정책이 이상하게 동작하거나 에러 | `--wrist-crop ""` |
| `EVAL_CUTOFF` 너무 타이트 | 파킹 전에 컷오프가 걸려 팔이 멈춘 채 방치될 뻔함 | 90 이상 권장(§4). 파킹 이후는 자동으로 컷오프 무시됨(2026-08-18 수정) |
| 카메라를 옮김 | 도형 검출 bbox가 어긋나거나 테이프 자국 배제 구역이 안 맞음 | §2로 재확인, 필요시 `--board`/`--exclude` 값 재측정 |
| 이물질(테이프 자국)을 실제로 뗐음 | `DEFAULT_EXCLUDE` 자리에 진짜 도형을 그려도 안 잡힘 | `ink_metric.py`/`erase_eval.py`/`erase_eval_ui.py` 호출에 `--exclude 0,0,0,0` 추가해서 기본 배제를 끈다 |
| `--mode demo`로 돌림 | 롤아웃이 기록 안 돼 나중에 재채점 불가 | 평가에는 항상 `--mode augment` |
| GUI 창을 여러 개 동시에 띄움 | 카메라 장치 충돌(`Device or resource busy`) | 한 번에 세션 하나만 |

---

## 참고

- 채점 지표 정의·통계 근거·실험 설계: [evaluation_protocol.md](evaluation_protocol.md)
- 추론 스무딩·안전 클램프 기본값: [smoothing.md](smoothing.md)
- 지우기 게이트/재시도 설계: [erase_run_design.md](../erase_run_design.md)
