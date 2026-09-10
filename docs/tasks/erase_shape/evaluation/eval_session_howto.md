# 실물 평가 세션 실행 설명서

지우기 평가 GUI(`erase_eval_ui.py`)로 실물 롤아웃을 돌리는 **실행 절차** 문서 —
"무엇을 어떻게 재는지"는 [erase_run_design.md](../runtime/erase_run_design.md)를 볼
것, 여기는 환경 세팅부터 명령까지만 다룬다.

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
python scripts/tasks/erase_shape/lib/ink_metric.py \
  <아무_기존_에피소드_폴더> --dump-roi /tmp/roi.png
```

`/tmp/roi.png`에서 볼 것:

- 보드 영역(파란 사각형)이 실제 화이트보드와 맞는지
- 그려진 도형에 bbox(빨간 사각형)가 정확히 씌워지는지
- **로봇 팔·나무 지우개 블록이 도형으로 잘못 잡히지 않는지**
- 노란 사각형(테이프 자국 배제 구역, `DEFAULT_EXCLUDE`)이 실제 테이프 자국
  위치와 맞는지 — 카메라를 옮겼거나 이물질을 실제로 뗐으면 §8 참고

안 맞으면 `--board X,Y,W,H`로 덮어써서 맞는 값을 찾은 뒤, 그 값을 아래 §4
명령의 `EVAL_BOARD`에 넣는다(생략하면 기본값 `228 0 515 720` 사용).

터미널에서 block 배치와 overlay view를 실행하고 싶으면:

```bash
conda activate ugrp
cd ~/UGRP/lerobot_robot_piper

python scripts/tasks/erase_shape/analysis/block_alignment_tool.py \
  --shape-zones 315 \
  --target-shape circle \
  --size-source global
```

**매 시도 전에 이 박스에 맞춰 도형을 그린다.** 위치·크기를 통일해야 시도끼리
`erased_frac`을 비교할 수 있다 — 기준(315 zone 6개, 도형별 median 크기)과 그 근거는
[erase_run_design.md §6](../runtime/erase_run_design.md)에 있다. 그린 뒤 `c`를 누르면
판정과 같은 파이프라인으로 검사해 `PASS`/`FAIL`이 뜬다.

> **132 체크포인트를 평가할 땐 `--shape-zones 132`**(4개)를 쓴다. 그 학습셋에는 315의
> 좌측 열에 해당하는 도형이 하나도 없어서, 거기 그리면 학습 분포 밖이 된다.

---

## 3. 체크포인트 선택

`outputs/train/<task>/<run_name>/checkpoints/<step 또는 last>/pretrained_model`
형태다. `last`가 보통 최종 스텝과 같다.

현재 있는 체크포인트(2026-08-19 기준, `smolvla` 매트릭스 전 칸 완료 — `ls outputs/train/*/*/`로
최신 목록 재확인할 것. pi0 top-only는 아직 학습된 적이 없어 표에 없다):

| 체크포인트 경로 | 종류 | wrist 카메라 | task 문장(`--task`) | 학습 데이터셋(`--dataset_root`) |
|---|---|---|---|---|
| `erase_the_shape/smolvla_erase_prompt_0812_0813am_72` | SmolVLA | 있음 | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_0812_0813am_72` |
| `erase_the_shape/smolvla_erase_prompt_toponly_0812_0813am_72` | SmolVLA | **없음(top-only)** | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_0812_0813am_72_top_only` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_topwrist_0812_0813am_72` | SmolVLA | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0812_0813am_72` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_toponly_0812_0813am_72` | SmolVLA | **없음(top-only)** | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0812_0813am_72_top_only` |
| `erase_the_shape/smolvla_erase_prompt_0727_0812_0813am_132` | SmolVLA | 있음 | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_0727_0812_0813am_132` |
| `erase_the_shape/smolvla_erase_prompt_toponly_0727_0812_0813am_132` | SmolVLA | **없음(top-only)** | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_0727_0812_0813am_132_top_only` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_132_v2` | SmolVLA | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_toponly_132` | SmolVLA | **없음(top-only)** | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132_top_only` |
| `pick_up_the_eraser_and_erase_the_shape/pi0_lora_pickup_prompt_132_decayfix3` | **pi0 LoRA** | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_132` | SmolVLA + **HAMLET** | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132` |
| `erase_the_shape/smolvla_erase_prompt_0802_0804_0805_0813pm_135` | SmolVLA | 있음 | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_0802_0804_0805_0813pm_135` |
| `erase_the_shape/smolvla_erase_prompt_toponly_0802_0804_0805_0813pm_135` | SmolVLA | **없음(top-only)** | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_0802_0804_0805_0813pm_135_top_only` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_topwrist_0802_0804_0805_0813pm_135` | SmolVLA | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0802_0804_0805_0813pm_135` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_toponly_0802_0804_0805_0813pm_135` | SmolVLA | **없음(top-only)** | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0802_0804_0805_0813pm_135_top_only` |
| `kwak/smolvla_return_prompt_0802_0804_0805_0813pm_135` | SmolVLA (3번째 프롬프트: return) | 있음 | `Pick up the eraser, erase the shape, and return the eraser to its original position.` | `records/outputs/kwak/pick_up_the_eraser_0802_0804_0805_0813pm_135` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_315` | SmolVLA | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315` |
| `pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_toponly_315` | SmolVLA | **없음(top-only)** | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_top_only` |
| `pick_up_the_eraser_and_erase_the_shape/pi0_lora_pickup_prompt_315` | **pi0 LoRA** | 있음 | `pick up the eraser and erase the shape` | `records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315` |
| `erase_the_shape/smolvla_topwrist_315_r1` | SmolVLA (erase만, pickup 없음) | 있음 | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_315` |
| `erase_the_shape/smolvla_toponly_315` | SmolVLA (erase만, pickup 없음) | **없음(top-only)** | `erase the shape` | `records/outputs/erase_the_shape/erase_the_shape_315_top_only` |

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

### HAMLET 체크포인트

HAMLET은 별도 Python policy 패키지(`smolvla_hamlet`)를 등록해야 checkpoint config를
읽을 수 있다. 현재 `smolvla_hamlet_pickup_prompt_132` 학습 출력에는 아직 완료된
`checkpoints/.../pretrained_model`이 없으므로, **학습 완료 후에만** 아래 절차를 쓴다.

```bash
# 현재 랩 PC의 HAMLET 소스 위치. 다른 위치에 뒀다면 그 경로로 바꾼다.
export PYTHONPATH=/home/ugrp43/jmbaek:${PYTHONPATH}
python -c "import smolvla_hamlet; print(smolvla_hamlet.__file__)"
```

§4의 `EVAL_ROLLOUT_CMD` 안에 다음을 추가한다.

```bash
--policy-discover-packages-path smolvla_hamlet
```

이 인자는 `erase_run.py`가 `piper_infer_runner.py`에 전달하며, policy를 로드하기
전에 `smolvla_hamlet`을 import한다. 일반 SmolVLA·π₀ 평가는 지정하지 않는다.

---

## 4. GUI로 세션 실행

```bash
EVAL_CUTOFF=120 \
EVAL_MODEL=<모델이름> \
EVAL_CONDITION=<조건이름> \
EVAL_TARGET=<circle|triangle|rectangle> \
EVAL_TRIALS=<목표 시행 수> \
EVAL_WATCH_DIR=records/rollout \
EVAL_ROLLOUT_CMD="python scripts/tasks/erase_shape/runtime/erase_run.py \
  --policy_path <체크포인트>/checkpoints/last/pretrained_model \
  --dataset_root <학습에 쓴 데이터셋 루트> \
  --task '<§3 표의 task 문장>' \
  --target <circle|triangle|rectangle> \
  --top-cam 327122074262 --wrist-cam 243322071626 \
  --top-crop 280,0,720 --wrist-crop 280,0,720 \
  --mode augment --stop-on-release \
  --confirm" \
bash scripts/13__eval_session.sh
```

### 인자 설명

| 인자 | 의미 |
|---|---|
| `EVAL_MODEL` / `EVAL_CONDITION` | 결과 분류용 라벨. 자유 문자열(예: `smolvla`/`pickup315`). 기본 결과 폴더 이름(`outputs/evaluation/erase_shape/<MMDD>_<model>_<condition>/`)에 들어간다 |
| `EVAL_OUT_DIR` | 기본 위치 대신 쓸 결과 폴더. 보통 지정하지 않는다 |
| `EVAL_TARGET` | 지금 보드에 그려진 도형과 일치시킬 것 |
| `EVAL_TRIALS` | 화면 표시용 목표 횟수일 뿐, 실제 반복 제어는 안 한다 — 원하는 만큼 space로 반복하면 된다 |
| `EVAL_WATCH_DIR` | **필수.** `erase_run.py --mode augment`가 새 에피소드를 만드는 상위 폴더. 항상 `records/rollout` |
| `EVAL_CUTOFF` | 초 단위 시간 제한. 기본 60인데 모델 로딩+2100스텝 추론(~70초)+파킹까지 합치면 크게 넘기므로 **120 이상 권장**. 파킹이 끝난 뒤(`[DISCONNECT]` 로그 이후)엔 후처리 예산으로 자동 전환되므로 너무 타이트하게 잡을 필요는 없다 |
| `--policy_path` | §3에서 고른 체크포인트의 `pretrained_model` 폴더 |
| `--policy-discover-packages-path` | HAMLET일 때만 `smolvla_hamlet`. 실행 전 해당 패키지의 상위 폴더를 `PYTHONPATH`에 넣는다 |
| `--dataset_root` | 그 체크포인트를 학습시킨 데이터셋(§3 표) |
| `--task` | **학습 때의 task 문장과 정확히 같아야 한다.** pickup 모델은 기본값과 같지만, erase-only 모델은 `erase the shape`를 명시한다 |
| `--top-crop` / `--wrist-crop` | 전 데이터셋이 `280,0,720`으로 통일돼 있다. **top-only 체크포인트면 `--wrist-crop ""`으로 비울 것** |
| `--mode augment` | 롤아웃을 LeRobotDataset으로 기록(크롭 전 원본 프레임 저장). 기록이 필요 없으면 `--mode demo` |
| `--stop-on-release` | 지우개를 확실히 놓으면(그리퍼가 파지 수준에서 뚜렷이 벌어짐) 즉시 파킹으로 간다. **항상 켜둘 것** — 없으면 정책이 다 끝내고도 남은 스텝(기본 2100)을 다 채운다. 놓지 않고 멈칫하는 시도는 안 끊고 `--max-steps`까지 간다(종료 사유가 `release`/`cutoff`로 갈리게) |
| `--max-steps` | 시도당 추론 스텝 상한. 기본 **2100**(시연 중앙값 720의 약 2.9배). 940 → 1410(약 2배)으로 올렸다가 다시 올렸다 — 1410을 다 쓰고도 마무리를 못 해 cutoff로 끝나는 시도가 반복됐다(2026-08-18 실물: 84.4% 지운 채 종료 — 꼭짓점 잔여). `--stop-on-release`가 켜져 있으면 다 놓는 즉시 끊기므로 상한을 올려도 성공하는 시도가 길어지지는 않는다 |
| `--confirm` | 실제로 팔을 움직인다. 빼면 인자 검증만 하고 종료(로봇 무동작) |

### 창 조작

| 키 | 동작 |
|---|---|
| space | 시작 / 중단 |
| Enter | 확인하고 다음 시행 |
| n | 이번 시도 버리기 (채점 안 됨) |
| 1~7 | 실패했으면 유형을 직접 고름(자동 추측 없음) |
| x | 이 에피소드를 집계에서 제외 |

⚠️ **[중단]은 안전장치가 아니다.** 프로세스를 종료할 뿐이라 팔이 즉시 서지
않을 수 있다. 위험하면 하드웨어 비상정지를 먼저 누르고, 그 다음 이 버튼으로
세션을 정리한다.

### 결과 확인

`outputs/evaluation/erase_shape/<MMDD>_<EVAL_MODEL>_<EVAL_CONDITION>/`에 쌓인다.

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

예: SmolVLA topwrist 132 vs 315 비교 (§3 표 기준 체크포인트).

```bash
# 조건 1: SmolVLA pickup 132 topwrist
EVAL_CUTOFF=120 \
EVAL_MODEL=smolvla EVAL_CONDITION=pickup132_topwrist \
EVAL_TARGET=<circle|triangle|rectangle> \
EVAL_WATCH_DIR=records/rollout \
EVAL_ROLLOUT_CMD="python scripts/tasks/erase_shape/runtime/erase_run.py \
  --policy_path outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_132_v2/checkpoints/last/pretrained_model \
  --dataset_root records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132 \
  --task 'pick up the eraser and erase the shape' \
  --target <circle|triangle|rectangle> \
  --top-cam 327122074262 --wrist-cam 243322071626 \
  --top-crop 280,0,720 --wrist-crop 280,0,720 \
  --mode augment --stop-on-release \
  --confirm" \
bash scripts/13__eval_session.sh

# 조건 2: SmolVLA pickup 315 topwrist
EVAL_CUTOFF=120 \
EVAL_MODEL=smolvla EVAL_CONDITION=pickup315_topwrist \
EVAL_TARGET=<circle|triangle|rectangle> \
EVAL_WATCH_DIR=records/rollout \
EVAL_ROLLOUT_CMD="python scripts/tasks/erase_shape/runtime/erase_run.py \
  --policy_path outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_315/checkpoints/last/pretrained_model \
  --dataset_root records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --task 'pick up the eraser and erase the shape' \
  --target <circle|triangle|rectangle> \
  --top-cam 327122074262 --wrist-cam 243322071626 \
  --top-crop 280,0,720 --wrist-crop 280,0,720 \
  --mode augment --stop-on-release \
  --confirm" \
bash scripts/13__eval_session.sh
```

top-only 조건으로 바꾸려면 `--policy_path`/`--dataset_root`를 `_toponly_`/`_top_only`
체크포인트로 바꾸고 `--wrist-cam ""  --wrist-crop ""`을 추가한다(§3 표 참고).

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
결과가 공정하다(페어링).

### 실험 운영 원칙

- **시행 횟수**: 조건당 10회(RTC/ForceVLA/Wall-OSS)~20회(SO-101)가 관행. 가능하면
  30~50회 — 이진 성공률만으로는 그래도 부족하다(연속 지표 `erased_target`을
  같이 볼 것).
- **페어링**: 같은 초기 조건(도형 위치·모양·팔 초기자세)을 모든 조건에 동일하게
  적용한다.
- **순서 효과**: 마커 건조·보드 잔여물·지우개 포화 등 → 조건 순서를 랜덤화하고
  청소 주기를 고정한다.
- **저장 공간**: 평가용은 top 카메라만 있으면 된다. 1280×720 top 영상 기준 60초당
  약 13.5MB — 90 에피소드면 약 1.2GB. `record_raw_frames`는 끌 것.

---

## 6. 채점 도구를 GUI 없이 직접 쓰기

이미 녹화된 롤아웃 폴더를 다시 채점하거나, 여러 개를 한 번에 채점할 때:

```bash
python scripts/tasks/erase_shape/evaluation/erase_eval.py \
  'records/rollout/<데이터셋명>_rollout_*' \
  --model smolvla --condition async \
  --out-dir outputs/evaluation/erase_shape/재채점
```

기존 CSV의 요약만 다시 만들 때:

```bash
python scripts/tasks/erase_shape/evaluation/erase_eval.py \
  --summarize outputs/evaluation/erase_shape/<폴더>/episodes.csv
```

### 손으로 채점하기 (카메라·세그멘테이션이 실패할 때)

사람이 눈으로 잴 수 있는 것만 기록한다.

```bash
cp outputs/evaluation/erase_shape/manual_sheet_template.csv outputs/evaluation/erase_shape/<폴더>/episodes.csv
```

예시 행은 지우고, 시행마다 한 줄씩 채운다(나머지 칸은 비워둔다):

| 칸 | 채우는 법 |
|---|---|
| `episode` | 아무 식별자 (`t01`, `t02` …) |
| `model` / `condition` | 조건 이름. 집계 축이므로 철자 통일 |
| `trial` | 1부터 |
| `target` | circle / rectangle / triangle |
| `valid` | 실험 사고면 `False` |
| `scored_by` | `manual` |
| `erased_target` | 눈금 4단계: 안 지움 `0` / 절반쯤 `0.5` / 거의 다 `0.9` / 완전 `1.0` |
| `success` | 완전히 지웠으면 `True` |
| `termination` | `release` / `cutoff` |
| `failure_mode` | 실패면 §4 표에서 하나 |
| `note` | 자유 기록 |

`erased_target`을 소수점까지 재려 하지 말 것 — 4단계 눈금이 사람이 재현할 수 있는
유일한 해상도다. 집계는 위와 같은 `--summarize` 명령으로 자동 채점분과 동일하게
처리된다(`scored_by` 칸으로 나중에 구분 가능).

---

## 7. 위치·도형별로 갈라 보기 (`ink_by_zone.py`)

채점기는 조건별 성공률과 평균만 낸다. 그런데 실물에서 잘 지우고 못 지우는 차이는
**도형 종류보다 보드 위 어느 자리인가**에서 훨씬 크게 갈린다 — 그 축은 채점 결과에
안 남아 있어서 이 도구로 따로 붙인다.

```bash
python scripts/tasks/erase_shape/analysis/ink_by_zone.py \
  outputs/evaluation/erase_shape/<폴더>/episodes.csv
```

산출물(기본 `outputs/analysis/erase_shape/ink_by_zone/<폴더 이름>/`):

| 파일 | 내용 |
|---|---|
| `erased_by_zone.png` | 도형 전체 합산. 6개 zone 중심에 진행률 링(지운 비율 평균) |
| `erased_by_zone_<도형>.png` | 도형별로 같은 그림 (원/삼각형/사각형) |
| `ink_by_zone.csv` | zone × 도형 지운비율 평균·SEM·성공수 |
| `clean_board.png` | 합성한 빈 보드 (다른 그림에 재사용 가능) |

동작 방식에서 알아둘 것:

- **위치는 CSV에 없다.** 각 에피소드의 `reference_frame.png`를 `detect_shapes()`로
  다시 읽어 target 도형 중심을 구하고, 가장 가까운 zone에 배정한다(채점과 같은
  파이프라인이라 기준이 어긋나지 않는다). 배정 거리 중앙값/최대를 항상 출력하므로
  값이 크면 배정을 의심할 것 — 실측 2026-08-21 54회는 중앙값 6px / 최대 18px였다.
- **배경은 따로 찍지 않는다.** 에피소드 reference 프레임들의 픽셀별 중앙값을 쓴다 —
  도형이 자리마다 다르므로 중앙값을 취하면 도형만 사라지고 빈 보드가 남는다.
- **`episodes.csv`의 `path`가 깨져도 된다.** 절대경로라 아카이브로 옮기면 전부
  깨지는데, 그때는 에피소드 이름으로 `--rollout-root`(기본 `records/`) 아래를
  다시 찾는다.
- **8/20 재설계 이전 CSV는 못 쓴다.** `erased_target`/`remaining_frac` 컬럼이 없는
  구 스키마라 `[skip]`으로 걸러진다(조용히 0%로 때우지 않는다).

읽는 법: 링이 많이 찰수록 잘 지운 것이고, 가운데 숫자가 그 zone의 지운 비율 평균이다.
칸당 표본이 적으면(예: 3회) 개별 칸 값은 흔들리니 zone 사이의 큰 차이만 해석할 것.

---

## 8. 알려진 함정

| 함정 | 증상 | 대응 |
|---|---|---|
| `conda activate ugrp` 안 함 | 종료 사유가 전부 `unknown(no_log)` | 환경 확인 후 재실행 |
| top-only 체크포인트에 `--wrist-crop` 그대로 둠 | 관측 형태 불일치로 정책이 이상하게 동작하거나 에러 | `--wrist-crop ""` |
| `EVAL_CUTOFF` 너무 타이트 | 파킹 전에 컷오프가 걸려 팔이 멈춘 채 방치될 뻔함 | 120 이상 권장(§4, max-steps 2100 기준). 파킹 이후엔 후처리 예산으로 자동 전환됨(2026-08-18 수정) |
| 카메라를 옮김 | 도형 검출 bbox가 어긋나거나 테이프 자국 배제 구역이 안 맞음 | §2로 재확인, 필요시 `--board`/`--exclude` 값 재측정 |
| 이물질(테이프 자국)을 실제로 뗐음 | `DEFAULT_EXCLUDE` 자리에 진짜 도형을 그려도 안 잡힘 | `ink_metric.py`/`erase_eval.py`/`erase_eval_ui.py` 호출에 `--exclude 0,0,0,0` 추가해서 기본 배제를 끈다 |
| `--mode demo`로 돌림 | 롤아웃이 기록 안 돼 나중에 재채점 불가 | 평가에는 항상 `--mode augment` |
| GUI 창을 여러 개 동시에 띄움 | 카메라 장치 충돌(`Device or resource busy`) | 한 번에 세션 하나만 |

---

## 참고

- 판정 구현 메모: [erase_run_design.md](../runtime/erase_run_design.md)
- 위치·도형별 분석 도구: `scripts/tasks/erase_shape/analysis/ink_by_zone.py` (§7)
- 추론 스무딩·안전 클램프 기본값: [smoothing.md](../../../policy/smoothing.md)
- HIL(사람 개입) 설계 — 지금 평가 세션에서는 안 씀: [hil_intervention_design.md](../runtime/hil_intervention_design.md)
