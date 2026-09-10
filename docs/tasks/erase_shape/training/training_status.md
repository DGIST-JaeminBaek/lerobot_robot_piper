# 학습 현황 스냅샷 (2026-09-03 갱신)

프롬프트(구/신규) × 카메라 구성(top+wrist/top-only) × 데이터셋 규모(72/132/135/315)를
조합해 가며 진행 중인 학습 캠페인의 현재 상태를 정리한다. 개별 학습의 상세 절차는
`method.md`(기본 파이프라인), `smolvla_finetuning.md`/`pi0_finetuning.md`(정밀도·
스케줄러 등 근거)를 볼 것. 이 문서는 **"지금 뭐가 되어 있고 뭐가 안 되어 있는지"만**
다룬다.

## 1. 왜 이 조합들을 도는가

150개(원본 4전략)로 `erase the shape` 프롬프트만 학습했을 때 지우개를 집는 것부터
실패하는 경우가 있었다. 프롬프트를 `pick up the eraser and erase the shape`로 바꾸고
강경묵/백재민 수집분(132개)으로 다시 학습하니 파지가 잘 됐다. 근데 그 사이 프롬프트·
데이터 규모·데이터 구성(수집자)이 동시에 셋 다 바뀌어서 뭐가 원인인지 원리적으로
구분이 안 된다 — 그래서 각 축을 하나씩 통제하며 여러 조합을 학습 중이다.

세 번째 프롬프트(`...and return the eraser to its original position.`)는 "다 지운
뒤에도 복귀를 안 하고 머무는" 문제를 언어로 직접 지시해서 완화되는지 보는 별도
탐색이다.

## 2. 데이터셋

| 이름 | 원본 날짜/수집자 | 에피소드 | 프레임 | joint4 보정 | 쓰인 프롬프트 |
|---|---|---:|---:|---|---|
| 72 | 0812/0813 오전 (강경묵) | 72 | 33,280 | 해당 없음 (0802/0804 미포함) | erase, pickup |
| 132 | 0727+0812+0813오전 (강경묵+백재민) | 132 | 65,265 | 해당 없음 | pickup |
| 135 | 0802+0804+0805+0813오후 (조성일+곽민준) | 135 | 71,909 | **보정본 사용 확인**(0802/0804 82개 전부 `*_joint4_corrected`) | pickup, return(신규 3번째) |
| (미학습) | 0811 (이동규+백재민) | 48 | — | 해당 없음 | — |
| 315 | 전체 (132+135+48, 위 세 그룹의 합) | 315 | 167,589 | **보정본 사용 확인**(82/315) | erase, pickup |
| augmented 276 | 합성 증강본 (`records/outputs/erase_shape_augmented_276`) | 276 | 203,438 | — | 도형별 3종 |
| augmented 276 trimmed | 위와 같은 276개를 구간 잘라낸 판 | 276 | 147,159 | — | 도형별 3종 |

joint4 보정은 팀원 Minjun이 외부(Windows)에서 에피소드별 초반 60프레임 median을
빼는 방식으로 처리했고(`records/0802_joint4_corrected`, `0804_joint4_corrected`),
315/135 매니페스트 둘 다 원본이 아니라 보정본을 가리키는 것을 직접 확인했다.

프롬프트만 다르고 나머지(영상/state/action)는 동일한 데이터셋은 재인코딩 없이
`scripts/piper/recording/retask_dataset.py`로 복사해서 만들었다(72↔erase, 315↔pickup, 135↔
return/kwak).

## 3. Step/epoch 컨벤션

`steps = epoch × frames / batch`, batch=8 고정이라 **8 epoch을 목표로 하면
steps ≈ frames**다.

| 데이터셋 | 프레임 | 사용한 steps | 실제 epoch | checkpoint 저장 간격 (`save_freq`) |
|---|---:|---:|---:|---|
| 72 | 33,280 | 33,280 | 8.00 | 4,160 (= steps/8, 8개 저장) |
| 132 | 65,265 | 65,265 | 8.00 | 5,000 (13~14개 저장) |
| 135 | 71,909 | 71,909 | 8.00 | 8,989 (= steps/8, 8개 저장) |
| 315 | 167,589 | **168,000** | 8.02 | SmolVLA는 15,000(12개 저장), **pi0는 5,000**(34개 저장) — 같은 315+pickup인데 아키텍처별로 간격이 다르다 |

315개만 정확한 8 epoch(167,589) 대신 168,000을 쓴 이유: 랩 PC에서 먼저 학습된 예전
315개 SmolVLA 런(`smolvla_toponly_315`, `smolvla_topwrist_315_r1`)이 이미 168,000으로
돌아가 있어서, 이후 315개 재학습들도 거기 맞춰 스텝 수 차이를 비교 변수에서 제거했다.

`save_freq`는 일관된 규칙이 없다 — 72/135는 `steps//8`, 132는 5,000 고정, 315는
SmolVLA가 랩 PC 기존 런(15,000)을 따라갔고 pi0는 그와 무관하게 5,000을 썼다. 특정
step의 중간 checkpoint를 찾을 때는 이 표로 간격을 먼저 확인할 것.

**모든 런에서 `--policy.scheduler_decay_steps`를 위 steps 값과 정확히 일치시켰다**
(불일치 시 LR이 조기에 floor로 떨어지는 lerobot 스케줄러 버그 — §5 참고).

## 4. 현재 완료 현황

### SmolVLA

| 데이터셋 | erase(구) top+wrist | erase(구) top-only | pickup(신) top+wrist | pickup(신) top-only |
|---|---|---|---|---|
| 72 | ✅ `erase_the_shape/smolvla_erase_prompt_0812_0813am_72` | ✅ `smolvla_erase_prompt_toponly_0812_0813am_72` | ✅ `.../smolvla_topwrist_0812_0813am_72` | ✅ `.../smolvla_toponly_0812_0813am_72` |
| 132 | ✅ `smolvla_erase_prompt_0727_0812_0813am_132` | ✅ `smolvla_erase_prompt_toponly_0727_0812_0813am_132` | ✅ `.../smolvla_pickup_prompt_132_v2` | ✅ `.../smolvla_pickup_prompt_toponly_132` |
| 135 | ✅ `smolvla_erase_prompt_0802_0804_0805_0813pm_135` | ✅ `smolvla_erase_prompt_toponly_0802_0804_0805_0813pm_135` | ✅ `.../smolvla_topwrist_0802_0804_0805_0813pm_135` | ✅ `.../smolvla_toponly_0802_0804_0805_0813pm_135` |
| 315 | ✅ `erase_the_shape/smolvla_topwrist_315_r1` | ✅ `erase_the_shape/smolvla_toponly_315` | ✅ `.../smolvla_pickup_prompt_315` | ✅ `smolvla_pickup_prompt_toponly_315` |

135개 전용 세 번째 프롬프트(return): ✅ `outputs/train/kwak/smolvla_return_prompt_0802_0804_0805_0813pm_135` (top+wrist만, top-only·다른 데이터셋 규모는 없음)

### pi0 (LoRA r=32/alpha=64, 스케줄러 정상인 것만 — top-only는 아직 시도 안 함)

| 데이터셋 | erase(구) | pickup(신) |
|---|---|---|
| 72 | ❌ | ❌ |
| 132 | ❌ | ✅ `pi0_lora_pickup_prompt_132_decayfix3` |
| 135 | ❌ | ❌ |
| 315 | ❌ | ✅ `pi0_lora_pickup_prompt_315` |

### HAMLET (메모리 모듈 추가, SmolVLA 기반 — SmolVLA 매트릭스와 별개 표)

**속도가 완전히 다르다.** wandb 실측(step/s):

| 카메라 | step/s | 일반 SmolVLA 대비 |
|---|---:|---|
| top+wrist | 3.6~3.7 | 약 2.5배 느림 |
| top-only | 5.2~5.5 | 약 1.7~1.8배 느림 |

| 데이터셋 | erase(구) top+wrist | erase(구) top-only | pickup(신) top+wrist | pickup(신) top-only |
|---|---|---|---|---|
| 132 | — | — | ✅ `smolvla_hamlet_pickup_prompt_132` (실물 파지 문제 진단·해결 — §7 참고) | — |
| 315 | ✅ `smolvla_hamlet_topwrist_315` | ✅ `smolvla_hamlet_toponly_315` | ✅ `smolvla_hamlet_pickup_prompt_315` (168,000 step 완료, decay 일치 확인) | ✅ `smolvla_hamlet_pickup_prompt_toponly_315` (168,000 step 완료, decay 일치 확인) |

HAMLET 학습 CLI는 일반 SmolVLA와 다르다 — `--policy.type=smolvla_hamlet
--policy.discover_packages_path=smolvla_hamlet`, `PYTHONPATH`에
`/home/ugrp43/jmbaek` 포함 필요, `--policy.input_features`를 카메라별로 명시
(`--policy.path` 대신 `--policy.pretrained_path=lerobot/smolvla_base` 사용). 근거·
전체 예시는 `/home/ugrp43/jmbaek/smolvla_hamlet/README.md` "lerobot-train CLI로 직접
돌리기" 절.

#### HAMLET memory_stride 스윕 (2026-08-27~28)

`memory_window=4` 슬롯을 몇 프레임 간격으로 뽑을지(`memory_stride`)만 바꾼 단일 변수
ablation. **데이터셋·steps·batch·chunk_size·시드까지 나머지는 전부 동일**하다.
`observation_delta_indices = [-(window-1)·stride, …, -stride, 0]`이므로 stride가
곧 "메모리가 과거 몇 초까지 닿는가"다(30Hz 기준).

| 체크포인트 | 데이터셋 | stride | 메모리 도달 | 최종 step | 상태 |
|---|---|---:|---:|---:|---|
| `smolvla_hamlet_pickup_prompt_315` | 315 | 1 | 0.10초 | 168,000 | ✅ (스윕 이전 기본값) |
| `smolvla_hamlet_pickup_prompt_315_stride16` | 315 | 16 | 1.6초 | 168,000 | ✅ |
| `smolvla_hamlet_pickup_prompt_315_stride35_v4` | 315 | 35 | 3.5초 | 168,000 | ✅ 실물 54회 평가함(0823) |
| `smolvla_hamlet_pickup_prompt_315_stride50` | 315 | 50 | 5.0초 | 168,000 | ✅ |
| `smolvla_hamlet_pickup_prompt_132_stride35` | 132 | 35 | 3.5초 | 65,265 | ✅ 실물 54회 평가함(0824) |
| `smolvla_hamlet_pickup_prompt_315_stride35_tcl` | 315 | 35 | 3.5초 | 168,000 | ✅ |

`_tcl`은 stride35에 **사전학습한 moment token을 얹은 변형**이다
(`--policy.load_moment_tokens_from=outputs/train/tcl/tcl_315/moment_tokens.pt`).
2026-08-28 14:53 시작, 168,000 step 완료(2026-09-03 확인).

⚠️ **학습과 실물 평가를 동시에 돌리지 말 것.** `--num_workers=8`이 16코어 중 약 10개를
가져가서 평가 러너의 제어 주기가 **29.0Hz → 25.4Hz(−18%)**로 떨어졌다(2026-08-28 실측).
학습 데이터가 30Hz 기준이라 그리퍼가 닫히는 데 걸리는 시간도 그만큼 길어져서, 그날
얻은 stride16/50 결과는 0823 stride35(29Hz)와 직접 비교할 수 없게 됐다. 평가 중에는
`kill -STOP <pid>` / `kill -CONT <pid>`로 학습을 잠시 멈추는 편이 낫다.

스윕 결과와 관찰된 실패 양상은 `../evaluation/`의 세션 기록을 볼 것. 요약하면
stride16은 **파지 실패**(그리퍼를 벌렸다 닫았다 반복, 방향전환 66~69회/분 vs 사람
데모 7.5회/분)가, stride35는 **조기 종료**(58%만 지우고 스스로 놓음)가 지배적이다.

### 선택성 실험용 데이터셋 (2026-08-24~25)

`erase the shape` 한 문장으로는 정책이 "어느 도형을 지울지"를 언어에서 배울 압력이
없다 — 보드에 도형이 하나뿐이라 프롬프트의 도형 단어가 시각 입력과 100% 중복이기
때문이다. 그 압력을 만드는 두 데이터셋을 파생시켰다. **둘 다 315개·167,589프레임으로
원본과 같고**, 관절 상태·행동은 손대지 않았다.

| 체크포인트 | 데이터셋 | 무엇이 다른가 | 최종 step |
|---|---|---|---:|
| `smolvla_pickup_315_shape_prompt` | `pick_up_the_eraser_315_shape_prompt` | task 문장을 도형별 3종으로 분리 (`… erase the circle/triangle/rectangle`) | 168,000 |
| `smolvla_pickup_315_distractor` | `pick_up_the_eraser_315_distractor` | 위 3종 프롬프트 + **top 영상에 방해 도형 하나를 합성** (타겟과 절대 안 겹침, 조합당 105개 균등) | 168,000 |

생성 스크립트: `scripts/tasks/erase_shape/dataset/make_shape_prompt_dataset.py`,
`make_distractor_dataset.py`. 후자는 top 영상만 다시 인코딩하고 나머지는 하드링크라
용량을 새로 먹지 않는다. 방해 도형은 **합성이므로 실물 평가 때는 사람이 보드에 직접
그려야 한다** — 정렬 도구를 타겟/방해 도형 각각으로 띄워 크기를 맞출 것
(도형별 median: circle 127×124, triangle 106×129, rectangle 118×119).

### ABP (B-spline flow, SmolVLA 기반 — HAMLET과 별개 계열)

ABPolicy를 SmolVLA에 이식한 변형이다. HAMLET이 *과거를 기억하는* 축을 건드리는 반면,
이쪽은 *행동을 어떻게 표현하는가*를 바꾼다 — action chunk를 스텝별 값이 아니라 B-spline
계수로 예측해 궤적을 매끄럽게 만든다.

| 체크포인트 | 데이터셋 | policy type | 주요 설정 | 최종 step |
|---|---|---|---|---:|
| `smolvla_abp_pickup_prompt_315` | 315 | `smolvla_abp` | `bspline_degree=3`, `action_history_horizon=8`, `chunk_size=50` | 168,000 ✅ |

원 논문: Yang et al., "ABPolicy: Asynchronous B-Spline Flow Policy for Real-Time and
Smooth Robotic Manipulation," ICRA 2026 (arXiv:2602.23901).

**실물 평가는 아직 안 했다.**

### 합성 증강 276 — 전체본 vs 구간 잘라낸 판

같은 276개 에피소드를 프레임 구간만 다르게 자른 두 데이터셋으로 각각 학습해, 잘라내기가
성능에 미치는 영향을 보려는 쌍이다. 두 GPU에서 동시에 돌렸다(이름의 `gpu0`/`gpu1`).

| 체크포인트 | 데이터셋 | 프레임 | 최종 step |
|---|---|---:|---:|
| `smolvla_erase_shape_augmented_276_gpu0` | `erase_shape_augmented_276` | 203,438 | 30,000 |
| `smolvla_erase_shape_augmented_276_trimmed_gpu1` | `erase_shape_augmented_276_trimmed` | 147,159 | 30,000 |

trimmed 쪽이 **56,279프레임(27.7%) 적다.** 두 데이터셋 모두 에피소드 수(276)와 task
3종은 같고 프레임 수만 다르다 — 무엇을 기준으로 잘랐는지는 이 저장소에서 확인되지
않았다(생성 스크립트 미확인). 30,000 step은 다른 학습(168,000)보다 훨씬 짧아 예비
비교 성격으로 보인다. **실물 평가는 아직 안 했다.**

## 5. 제외된 체크포인트 (`outputs/train/trash/`)

lerobot의 `scheduler_decay_steps`가 `--steps`를 안 따라가는 버그(기본값 30000에 고정,
`--policy.scheduler_decay_steps`로 명시해야만 정상)에 걸려서 학습 후반부 LR이 조기에
바닥난 채로 끝난 체크포인트들. 상세 원인·재발 방지 체크리스트는
`pi0_finetuning.md`의 "LR 스케줄러" 절 참고.

- `pi0_lora_topwrist_75k` (75k, decay=30000)
- `pi0_lora_topwrist_315` (168k, decay=30000 — `--scheduler.num_decay_steps=67200`을 줬지만 무시됨)
- `pi0_lora_pickup_prompt_132` (65265, decay=30000)
- `pi0_lora_pickup_prompt_132_decayfix` (65265, decay=30000 — 첫 수정 시도 실패)
- `smolvla_erase_shape_0812_0813_morning` — 스케줄러 문제 아니고, 프롬프트를 잘못 넣고 학습한 실패작(사용자 확인)

## 6. HAMLET 추론 버그 — 메모리 시간축 불일치 (진단·해결, 2026-08-19)

`smolvla_hamlet_pickup_prompt_132`를 실물 Piper에 처음 연결했을 때, **기준선(무-메모리)은
지우개를 잘 집는데 HAMLET만 끝부분만 물거나 놓치고 거치대 근처만 계속 문지르는** 증상이
있었다. 학습 loss/grad_norm은 기준선과 동일, 아키텍처 용량도 동일, 세션 시작 메모리
리셋도 정상 — 원인은 **rollout 시 메모리 FIFO가 "정책 호출 횟수"로 시간을 재는데, 실제
호출 간격이 학습 때(연속 33ms 프레임)보다 훨씬 길어서 학습/추론 시간축이 어긋나는 것**으로
좁혀졌다. 학습 문제가 아니라 순수 추론 파이프라인 버그였다.

`/home/ugrp43/jmbaek/smolvla_hamlet/` 쪽에 이 진단을 공유했고, **재학습 없이 재추론
파이프라인만 고쳐서 해결됐다** — 로봇이 30fps(학습 fps)로 실제 카메라 프레임을 타임스탬프와
함께 계속 버퍼에 쌓아두고(`scripts/piper/inference/hamlet_history.py`, 신규), 정책 호출
시점에 "진짜 33ms 간격 4프레임"을 그 버퍼에서 재구성해서 넘긴다. `smolvla_hamlet` 패키지
쪽 `predict_action_chunk(..., rollout_history=...)`가 이걸 받으면 학습 때 쓰던 진짜
K-frame 처리 경로를 그대로 재사용한다(근사가 아니라 학습과 동일한 계산). 기존
`memory_stride=1` 체크포인트 그대로 재사용 가능 — **재학습 불필요.**

`piper_infer_runner.py`, `piper_human_approved_inference.py` 둘 다 이 버퍼가 배선돼 있다.
상세 진단·해결 과정, 실행 방법, `memory_stride` 재학습 실험은 전부
`/home/ugrp43/jmbaek/smolvla_hamlet/docs/ARCHITECTURE_DIFFERENCES.md`의 "부록: 실물 배포에서
드러난 설계 실수와 해결" 참고(이전에 나눠져 있던 `PIPER_MEMORY_STRIDE_INCIDENT.md` /
`PIPER_REAL_HISTORY_ROLLOUT.md`는 그 부록으로 통합됨). `memory_stride=35/50` 재학습은
그 뒤 실제로 완료했고(부록 A.4), `16`은 아직 후보로만 남아있다(부록 A.7).

## 7. 실물 평가

grasp는 위 §6 수정(real-history rollout, stride=1) 이후로 **모델 전부에서 성공하는
것으로 확인**되어 더 이상 비교축이 아니다. 어떤 체크포인트가 실제로 더 잘 지우는지에 대한
**정량적 실물 비교는 아직 하나도 없다.** 이 프로젝트의 잉크
기반 판정 도구(`erase_run.py` 게이트 시스템)는 아직 미완이라, 당장은 "같은 도형·같은
위치로 고정한 환경에서 모델만 바꿔가며 눈으로 비교"하는 방식으로 진행하기로 했다(B=132개
pickup 기준 vs C=315개 pickup부터).

## 8. lerobot SmolVLA 액션 패딩 loss 버그 (upstream, 2026-08-25 확인)

**지금까지 학습한 SmolVLA 계열 체크포인트는 전부(기준선 SmolVLA + HAMLET 전부) 이 버그
하에서 학습됐다.** 폐기 사유는 아니고(§5와 다름), 알고 있어야 할 조건이다.

### 무엇이 문제인가

lerobot의 `policies/smolvla/modeling_smolvla.py`가 액션 패딩 마스크를 **존재하지 않는
키**로 읽는다:

```python
actions_is_pad = batch.get("actions_id_pad")   # LeRobotDataset은 "action_is_pad"를 만듦
```

`"actions_id_pad"`는 lerobot 저장소 전체에서 이 한 줄에만 있고 만드는 곳이 없다. `.get()`
이라 에러도 경고도 없이 `None`이 되고, 바로 다음 `if actions_is_pad is not None:`이 False라
**패딩 마스킹이 조용히 건너뛰어진다.** ACT/Diffusion/TDMPC는 `action_is_pad`를 올바로 쓴다
(pi0/pi0fast는 애초에 패딩 마스킹 자체를 안 한다 — 이 버그와 무관).

### 실제 영향

액션 청크가 에피소드 끝을 넘어가면 `LeRobotDataset`은 인덱스를 클램핑한다
(`lerobot_dataset.py`, `max(ep_start, min(ep_end-1, idx+delta))`) — 즉 넘어간 칸은
**마지막 실제 액션의 복사본**으로 채워지고 `action_is_pad=True`로 표시된다. 마스킹이 안
걸리니 그 가짜 칸이 정답으로 채점됐고, 결과적으로 모델은 **"에피소드 끝 근처에서는 마지막
자세를 계속 유지하라"**를 학습했다.

`pick_up_the_eraser_315` 기준 규모(chunk_size=50, 중앙값 523프레임):

| | 값 |
|---|---:|
| 전체 액션 슬롯 | 8,379,450 |
| 그중 클램핑된(가짜) 슬롯 | 385,875 (**4.61%**) |
| 영향받는 학습 샘플 | 15,435 / 167,589 (**9.21%**) |

전체로는 4.6%지만 **에피소드 끝 49프레임에 몰려 있다** — 최종 프레임 샘플은 50칸 중 49칸
(98%)이 가짜다.

**단, 실제 성능 영향은 크지 않을 수 있다** — upstream PR에서 실제 학습으로 검증한 기여자
보고는 *"performance appears similar/marginally better"*다. §6의 grasp 실패 증상과는
무관하다(기준선도 같은 버그를 공유했는데 그 증상이 없었으므로).

### 왜 우리 버전에 남아있나

```
upstream 수정  f311ca3d "Fix action padding key at SmolVLA (#1717)"   2026-03-12
우리 체크아웃  8fff0fde                                                2026-02-27
```

우리 lerobot 체크아웃이 수정보다 약 2주 앞선다. 관련 이슈: [#1707], [#1717](수정),
[#3434](후속 — 마스킹 후 분모까지 유효 개수로 나눠야 loss가 안 줄어드는 문제).

### 대응 현황 — **판단 보류, 현재는 고치지 않음**

한 번 `smolvla_hamlet` 쪽에 수정을 적용했다가 **의도적으로 되돌렸다.** 지금은 HAMLET·기준선
둘 다 lerobot 0.4.4 원본과 동일한(버그 포함) 상태다. `modeling_smolvla_hamlet.py`의 해당
줄에 `DELIBERATE:` 주석을 달아뒀으니 모르고 다시 "고치지" 말 것.

**되돌린 이유 — 고칠 수 있는 쪽이 한쪽뿐이라서:**

| | 수정 가능 여부 |
|---|---|
| HAMLET(`smolvla_hamlet`) | 자체 `forward`를 복사해 갖고 있어 **가능** |
| 기준선 SmolVLA(`--policy.type=smolvla`) | lerobot 원본을 그대로 타므로 **불가** |

> ⚠️ HAMLET만 고치면 **loss 정의가 양쪽에서 달라져 HAMLET-vs-기준선 비교가 무의미해지고**,
> 이미 학습된 체크포인트 4개(stride 1/35/35/50)와도 비교가 안 된다. 지금 대기 중인 stride
> 비교 실험이 정확히 그 비교라, **그게 끝나기 전에는 손대면 안 된다.**

**선택지 (아직 결정 안 됨):**

1. **(현재) 그대로 둔다** — 모든 체크포인트가 같은 조건이라 비교가 공정. 대신 4.61%의 잘못된
   학습 신호를 계속 안고 간다.
2. **lerobot을 `f311ca3d` 이후로 업그레이드** — 양쪽이 동시에 고쳐져 가장 깨끗하다. 대신 그
   사이 다른 변경들이 딸려 와 통제 변수가 늘어난다.
3. **기준선도 얇은 서브클래스로 감싸 같은 수정 적용** — lerobot을 안 건드리고 양쪽을 맞춤.
   통제 변수를 가장 적게 흔들지만 기준선용 패키지를 새로 만들어야 한다.

**어느 쪽을 고르든, 고치는 순간 기존 4개 체크포인트는 비교 대상에서 빠지고 재학습이 필요하다.**
상세 근거·수치는 `/home/ugrp43/jmbaek/smolvla_hamlet/docs/ARCHITECTURE_DIFFERENCES.md` 부록 A.8.

### 나중에 고칠 때 주의할 것

**실물 롤아웃에는 영향 없다.** 이건 loss 계산식이지 파라미터가 아니고, `actions_is_pad`는
학습용 `forward()`에서만 쓰인다 — 추론
(`select_action`/`predict_action_chunk`/`sample_actions`)은 이 경로를 안 탄다. 그래서 기존
체크포인트 로드·실행은 수정 여부와 무관하게 그대로다.

**고칠 때는 두 가지를 같이 적용해야 한다.** 키만 고치면(#1717) 가짜 칸은 빠지지만 분모가
여전히 50이라, 패딩이 많은 에피소드 끝 샘플의 loss가 그만큼 작아져 gradient 기여가 줄어든다
(50칸 중 30칸 패딩이면 정상의 0.40배, 최종 프레임 샘플은 0.02배). 분모도 유효 개수로
바꿔야(#3434) 정상이 된다.

**loss 값 비교 금지**: 수정 후 잰 loss는 기존 wandb 로그값과 정의가 다르다(가짜 칸 포함
여부 + 분모). 체크포인트끼리 비교할 땐 반드시 **전부 같은 코드로** 재야 한다.

[#1707]: https://github.com/huggingface/lerobot/issues/1707
[#1717]: https://github.com/huggingface/lerobot/pull/1717
[#3434]: https://github.com/huggingface/lerobot/issues/3433
