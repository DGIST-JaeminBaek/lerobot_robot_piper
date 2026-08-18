# eval_kit — 판정 분리 + 에피소드 CSV

지우기 성능을 **게이트와 독립적으로** 다시 판정하고, 에피소드당 1행 CSV로 쌓는
평가 전용 모듈. 논문 §Experiments의 숫자는 전부 여기서 나오게 하는 것이 목적이다.

## 왜 분리하는가

`erase_check.py`는 게이트가 "이제 그만해도 되나"를 정하는 코드다. 그 코드로 성능도
판정하면 **"게이트가 성공이라 믿을 때 성공"** 이라는 동어반복이 된다. 게이트가
과대추정하는 방향으로 편향돼 있으면 성능이 그만큼 공짜로 부풀고, 그걸 검출할 방법이
없다. 리뷰어가 정확히 이 지점을 친다.

그래서 `judge.py`는 `scripts/tools`의 어떤 모듈도 import하지 않는다. 도형 찾기,
잉크 측정, 임계값까지 전부 독립적으로 정의돼 있고 임계값은 `judge_config.json`에
동결돼 있다. 두 경로가 어긋나면 CSV의 `disagree` 칼럼에 그대로 드러난다.

> 실측: 기존 저장된 실행 2건에서 독립 판정과 게이트 판정이 `target_erased` 소수
> 4자리까지 일치했다(0.4286 / 0.9905). 지금은 두 경로가 같은 답을 낸다는 뜻이고,
> 앞으로 갈라지면 그 순간을 CSV가 잡는다.

## 기존 코드에 대한 영향: 없음

이 폴더 밖의 파일은 **한 줄도 수정하지 않았다.** `erase_run.py`, `erase_check.py`,
`erase_stats.py`, `13__erase_gate.sh` 전부 그대로다. 입력은 `erase_run.py`가 이미
남기고 있는 산출물뿐이고, 전부 사후(offline) 처리다 — 로봇도 카메라도 정책도 필요 없다.

**지우는 법:**

```bash
rm -rf scripts/tools/eval_kit
rm -f records/hil/*/eval_kit.json
rm -rf records/hil/*/eval_kit
rm -f outputs/analysis/episodes.csv
```

이걸로 완전히 원래 상태다.

## 쓰는 순서

### 1. 실행 직후 — 도장 찍기 (재현성)

`meta.json`에 없는 정보(커밋, 조건, 보드 패턴 번호, 담당자)를 사이드카로 남긴다.

```bash
python scripts/tools/eval_kit/stamp.py --latest records/hil \
    --condition gate --pattern-id P03 --operator seongil
```

`--pattern-id`는 **페어링 통계(McNemar)의 전제**다. 같은 낙서 패턴을 모든 조건에
동일하게 제시하고 그 번호를 여기 적어야, n=20에서도 조건 비교가 유의를 잡는다.

### 2. 재판정 + CSV 누적

```bash
python scripts/tools/eval_kit/adjudicate.py records/hil/*
```

* `<run_dir>/eval_kit/verdict.json` — 판정 전문(도형 bbox, 잉크 원시값 포함)
* `outputs/analysis/episodes.csv` — 에피소드당 1행

`(run_id, attempt)` 키로 upsert하므로 몇 번을 다시 돌려도 행이 중복되지 않는다.
임계값을 바꿨으면 `--force`로 전부 다시 판정한다.

### 3. 표 뽑기

```bash
python scripts/tools/eval_kit/summarize.py --markdown
```

성공률 + Wilson 95% CI, `target_erased` 평균±sd, 훼손률, 위반/타임아웃 건수,
게이트 불일치 건수. 조건 간 유의성 검정과 효과 크기는 기존 `erase_stats.py`에 이미
있으니 거기서 본다 — 같은 걸 두 번 구현하지 않았다.

### 4. theta 캘리브레이션 (한 번, 실물 불필요)

```bash
python scripts/tools/eval_kit/calibrate.py --make-sheet 40 --runs 'records/hil/*'
# 3명이 따로 채운다 (rater_A/B/C에 1 또는 0)
python scripts/tools/eval_kit/calibrate.py --analyze outputs/analysis/labels.csv
```

라벨러 간 kappa, ROC/Youden 최적 theta, 현재 theta의 FP/FN을 낸다. 결과를 보고
`judge_config.json`의 `theta`를 **한 번만** 고치고 동결한다. 이 절차가 없으면
"theta를 왜 0.9로 뒀나"에 답할 수 없다.

## CSV 칼럼

| 그룹 | 칼럼 |
|---|---|
| 신원 | `run_id`, `attempt`, `is_final`, `run_dir` |
| 재현성 | `condition`, `pattern_id`, `operator`, `commit`, `dirty`, `started`, `policy_path`, `dataset_root`, `task`, `target`, `mode`, `hil`, `aggregate_fn`, `max_relative_target`, `top_crop`, `wrist_crop`, `judge_version`, `config_hash`, `adjudicated_at` |
| 성과 | `judge_success`, `target_erased`, `remaining_frac` |
| 위반·부작용 | `shape_damage`, `damage_violation`, `target_incomplete`, `worst_distractor` |
| 실행 특성 | `steps`, `max_steps`, `timeout`, `interventions`, `measured_fps`, `runner_status`, `aborted` |
| 게이트 대조 | `gate_success`, `gate_target_erased`, `disagree` |

`success` 정의는 `target_erased >= theta AND shape_damage <= max_damage`다. **방해
도형을 지워버린 시행은 target을 다 지웠어도 성공이 아니다** — 부작용을 성공에 섞으면
안 된다.

칼럼을 추가할 때는 `adjudicate.py`의 `COLUMNS` **뒤에만** 붙일 것. 순서를 바꾸면
기존 CSV와 이어붙지 않는다.

## 알려진 한계

* **판정 프레임은 1장이다.** 표준은 3장 중앙값인데, 그러려면 `erase_run.py`의 촬영
  경로를 고쳐야 해서 지금은 안 했다. 대신 자동 노출을 끄고(고정 노출) 찍는 것으로
  프레임 노이즈를 줄이는 게 더 싸다.
* **도형 위치는 기준 프레임에서만 찾는다.** 시도 중 보드가 밀리면 bbox가 어긋난다.
  실행 중 보드를 건드리지 말 것.
* **`condition`/`pattern_id`는 사이드카가 있어야 채워진다.** 안 찍고 돌린 실행은
  나중에 `stamp.py <run_dir> --condition ...`으로 소급해서 붙일 수 있다.

## 테스트

```bash
python scripts/tools/eval_kit/tests/test_eval_kit.py
```

합성 이미지로 돈다 — 로봇도, 저장된 실행도 필요 없다.
