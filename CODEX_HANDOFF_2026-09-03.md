# Codex 작업 인수인계 — erase-shape / SmolVLA

최종 갱신: 2026-09-03 (Asia/Seoul)

이 문서는 다른 Codex 세션/계정이 지금까지의 대화 맥락과 작업 상태를 로컬 워크스페이스에서 이어받기 위한 요약이다. 대화 원문 전체가 아니라, 실행에 필요한 결정·경로·검증 결과·주의사항을 구조화해 기록한다.

## 최우선 사용자 요구

- 기존 데이터셋, 기존 모델, 기존 스크립트를 임의로 수정하거나 덮어쓰지 않는다.
- 새 작업은 별도 파일과 별도 출력 경로로 만든다.
- 현재 새 학습은 **물리 GPU 1번을 강제 사용**해야 한다.
- 장시간 작업은 tmux에서 실행하고, 사용자가 `tail -F`로 볼 수 있는 로그를 남긴다.
- 진행 상황 질문에는 불필요한 전체 로그를 읽지 말고 `n/전체`, 경과시간, ETA만 빠르게 답한다.
- 모델 계열은 SmolVLA이다. HAMLET이나 Pi0를 결합하는 작업이 아니다.

## 워크스페이스와 환경

```text
프로젝트: /home/ugrp43/UGRP/lerobot_robot_piper
conda 환경: ugrp
Python 환경 경로: /home/ugrp43/miniconda3/envs/ugrp
```

기본 진입:

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper
```

주의: 이 저장소의 worktree에는 사용자 소유의 수정/삭제/미추적 파일이 매우 많다. `git reset`, `git checkout --`, 광범위한 삭제나 정리 작업을 하지 않는다.

## 데이터 준비 흐름

### 원본 276개 증강 세트

- 작업 폴더: `records/augmented_test`
- 원래 `records`에서 증강본이 존재하는 범위만 복사했다.
- 합성 영상은 top camera만 교체했고 wrist와 비영상 데이터는 해당 원본에서 가져왔다.
- 총 276 episode이며 task별 92개씩이다.
- task 문자열은 아래 세 가지다.

```text
pick up the eraser and erase the circle
pick up the eraser and erase the triangle
pick up the eraser and erase the rectangle
```

관련 파일:

```text
records/augmented_test/build_erase_shape_augmented_276.py
records/augmented_test/erase_shape_augmented_276_manifest.json
```

초기 빌드에서 확인했던 문제:

1. 동적 import한 dataclass 모듈이 `sys.modules`에 없어 `AttributeError` 발생.
2. manifest 항목에 `_source_path`가 없어 `KeyError` 발생.
3. encoder queue가 차서 top/wrist frame drop 경고 발생.
4. 이 문제들을 기존 빌더를 직접 바꾸는 방식이 아니라 `records/augmented_test` 아래의 별도 빌드 파일에서 처리했다.

초기 untrimmed 출력:

```text
records/outputs/erase_shape_augmented_276
records/outputs/.erase_shape_augmented_276_single_task
```

untrimmed 빌드 결과는 276 episodes, 203,438 frames였고, task별 circle/triangle/rectangle 92개로 매핑했다. 이후 실제 동작 시작 전 긴 정지/움찔 구간이 학습에 악영향을 준다고 판단하여 QC 구간 JSON을 적용한 trimmed 데이터셋을 새로 만들었다.

### 현재 사용 중인 trimmed 276 데이터셋

관련 파일:

```text
records/augmented_test/build_erase_shape_augmented_276_trimmed.py
records/augmented_test/erase_shape_augmented_276_trimmed_manifest.json
```

최종 데이터셋:

```text
/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_shape_augmented_276_trimmed
```

검증된 수치:

- episodes: 276
- tasks: 3, 각 92 episodes
- frames/parquet rows: 147,159
- 카메라: top + wrist, 512×512
- video shards: 총 7개(top 4개, wrist 3개). episode당 MP4 하나가 아니라 여러 episode가 shard MP4에 연속 저장되는 LeRobot 구조다.
- top decoded frames: 147,159 / expected 147,159
- wrist decoded frames: 147,159 / expected 147,159
- 완전 디코딩 및 parquet/video 프레임 정렬 검증 통과

7개 MP4:

```text
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.top/chunk-000/file-000.mp4
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.top/chunk-000/file-001.mp4
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.top/chunk-000/file-002.mp4
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.top/chunk-000/file-003.mp4
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.wrist/chunk-000/file-000.mp4
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.wrist/chunk-000/file-001.mp4
records/outputs/erase_shape_augmented_276_trimmed/videos/observation.images.wrist/chunk-000/file-002.mp4
```

빌드 완료 당시 로그 핵심:

```text
영상 7개 링크 완료
data parquet 147159행 task_index 갱신
meta/episodes 1개 파일 갱신
meta/info.json(total_tasks), stats.json(task_index) 갱신
PASS: created .../records/outputs/erase_shape_augmented_276_trimmed
BUILD_EXIT_STATUS=0
```

임시 single-task 데이터셋도 보존되어 있다.

```text
records/outputs/.erase_shape_augmented_276_trimmed_single_task
```

## 현재 진행 중인 새 SmolVLA 학습

### 목적

기존 315 데이터 학습의 주요 기준(batch 8, 약 8 epochs)을 trimmed 276 데이터셋에 맞춰 적용한다. 147,159 frames에서 batch size 8로 `steps=147159`이면 sample 기준 정확히 8 epochs다.

### 전용 실행 스크립트

```text
/home/ugrp43/UGRP/lerobot_robot_piper/scripts/training/train_gpu1_smolvla_erase_shape_augmented_276_trimmed_8ep.sh
```

이 파일은 이번 작업을 위해 새로 만든 별도 파일이다. 기존 학습 스크립트나 모델을 수정하지 않았다.

주요 설정:

```text
model: SmolVLA (`lerobot/smolvla_base`)
physical GPU: 1 (`CUDA_VISIBLE_DEVICES=1`)
policy device: cuda
push_to_hub: false
dataset: records/outputs/erase_shape_augmented_276_trimmed
batch_size: 8
num_workers: 2
steps: 147159
effective epochs: 8.0
seed: 1000
log_freq: 100
save_freq: 15000
eval_freq: 0
chunk_size: 50
n_action_steps: 50
optimizer lr: 1e-4
weight decay: 1e-10
grad clip norm: 10
warmup steps: 1000
cosine decay steps: 147159
decay lr: 2.5e-6
freeze vision encoder: true
train expert only: true
train state projection: true
W&B project: smolvla-erase-shape
W&B run id: smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1
```

출력은 새 경로다.

```text
/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1
```

로그:

```text
/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1.log
```

tmux session:

```text
train_smolvla_aug276_trimmed_8ep_gpu1
```

시작에 사용한 명령:

```bash
tmux new-session -d -s train_smolvla_aug276_trimmed_8ep_gpu1 'bash /home/ugrp43/UGRP/lerobot_robot_piper/scripts/training/train_gpu1_smolvla_erase_shape_augmented_276_trimmed_8ep.sh'
```

실시간 로그:

```bash
tail -F /home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1.log
```

tmux 접속:

```bash
tmux attach -t train_smolvla_aug276_trimmed_8ep_gpu1
```

접속 상태에서 빠져나오기: `Ctrl-b`, 이어서 `d`.

W&B:

```text
https://wandb.ai/jmbaek-daegu-gyeongbuk-institute-of-science-technology/smolvla-erase-shape/runs/smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1
```

2026-09-03 20:47 KST에 정상 시작했다. 시작 확인 당시:

- dataset 147,159 frames / 276 episodes 인식
- effective batch size 8
- 학습 가능한 파라미터 약 100M / 전체 약 450M
- GPU 1에서 PID 200452, 약 3.9 GiB compute memory 사용
- GPU 1 전체 약 4.45 GiB, utilization 약 89%
- GPU 0은 약 15 MiB, utilization 0%
- step 100 로그: loss 0.296, grad norm 2.849
- 초기 실측 속도 약 7.5 steps/s, 당시 ETA 약 5시간 20분

첫 시작 시 `--policy.push_to_hub=false`가 없어 설정 검증 단계(step 0)에서 한 번 즉시 종료됐다. 실제 학습이나 체크포인트 생성 전의 실패였으며, 전용 스크립트에 아래를 명시한 뒤 재시작해 정상 동작 중이다.

```text
--policy.device=cuda
--policy.push_to_hub=false
```

GPU 확인:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

학습 종료 확인은 로그 마지막의 `TRAIN_EXIT_STATUS=0`과 `checkpoints/last/pretrained_model` 존재를 함께 확인한다.

## 이전 모델과 평가에서 얻은 교훈

이전에 `outputs/train/smolvla_erase_shape_augmented_276_trimmed_gpu1`에서 30,000-step 학습을 수행했으나 약 1.63 epochs에 불과했다. 새 학습은 이를 덮어쓰지 않고 `_8ep_gpu1` 경로에 별도로 생성한다.

untrimmed 276 모델 테스트에서는 로봇이 기본 자세에서 500~900 steps가량 움찔거리다가 늦게 움직이는 현상이 있었다. 데이터 수집 영상의 길이가 서로 다른 것 자체는 문제가 아니지만, 각 episode 앞부분의 긴 무동작 구간은 학습 분포를 왜곡한다. 그래서 QC cut range를 반영한 trimmed 276 데이터셋을 다시 빌드하고 현재 재학습 중이다.

평가 실행 시 Hugging Face에 `HEAD` 요청을 하다가 DNS/timeout 재시도가 발생했다. 모델 파일은 로컬에 있으므로 필요하면 아래 환경변수를 평가 명령 앞에 둔다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

정책 경로는 줄바꿈으로 두 토큰이 되면 안 된다. 반드시 다음처럼 하나의 연속된 경로여야 한다.

```text
outputs/train/<run>/checkpoints/last/pretrained_model
```

현재 trimmed 8-epoch 모델이 끝난 뒤 GUI 기반 평가를 구성할 때의 핵심 rollout 형식:

```bash
python scripts/tasks/erase_shape/runtime/erase_run.py --policy_path outputs/train/smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1/checkpoints/last/pretrained_model --dataset_root records/outputs/erase_shape_augmented_276_trimmed --task 'pick up the eraser and erase the circle' --target circle --top-cam 327122074262 --wrist-cam 243322071626 --top-crop 280,0,720 --wrist-crop 280,0,720 --mode demo --max-steps 900 --stop-on-release --confirm
```

shape별로 `--task`와 `--target`을 반드시 일치시킨다.

평가/도달 세션 wrapper:

```text
scripts/13__eval_session.sh
scripts/14__reach_session.sh
```

`REACH_CUTOFF` 또는 `EVAL_CUTOFF`는 wrapper가 wall-clock 기준으로 프로세스를 종료시키는 값이다. 모델 최초 로딩/Hugging Face timeout 시간도 포함되므로 너무 짧으면 실제 rollout 초반에 잘릴 수 있다. 과거 60초 cutoff에서 모델 로딩 후 134 steps만 수행하고 종료된 사례가 있었다.

## 카메라/로봇 관련 값

```text
top camera serial: 327122074262
wrist camera serial: 243322071626
top crop: 280,0,720
wrist crop: 280,0,720
CAN leader: can_leader
CAN follower: can_follower
```

GUI 실행:

```bash
cd /home/ugrp43/UGRP/lerobot_robot_piper
bash scripts/0__launch_gui.sh
```

한때 GUI에서 leader/follower 관절값이 모두 0으로 고정되었고 `scripts/1__init_can.sh`에서 named interface가 없다는 경고가 발생했다. USB 재연결/로봇 재부팅 후 CAN interface 재초기화를 점검했다. 이 상태는 하드웨어 의존적이므로 향후 녹화나 평가 전 반드시 `ip -details link show` 또는 GUI CAN monitor에서 실제 관절값이 변하는지 재확인한다.

## 2026-09-02 실데이터 20개

circle target 실데이터 20개가 아래에 있다.

```text
/home/ugrp43/UGRP/lerobot_robot_piper/records/0902
```

각 폴더 이름은 대체로 다음 형식이다.

```text
pick_up_the_eraser_and_erase_the_circle_0902-HHMMSS
```

QC JSON:

```text
records/0902/qc_review_0902_circle_20.json
```

대화 중 `0902-195334`부터 `0902-200003`까지의 세 episode가 모두 약 2.4초로 사용 불가 표시된 것이 잘못됐다고 사용자가 지적했고 수동 마킹을 다시 확인했다. 이 실데이터를 새 학습에 합치기 전에는 해당 JSON의 세 항목과 실제 선택 프레임 범위를 다시 검증한다.

향후 아이디어로, 실제 distractor 환경 데이터를 shape별 20개(예: circle target + triangle distractor 10개, rectangle distractor 10개), 총 60개 수집하고 대량 증강 데이터와 혼합/단계 학습하는 방안을 논의했다. 아직 이 60개 전체를 만든 것은 아니다.

## joint4 보정 관련 미확정 사항

사용자가 `joint4_corrected_review` 및 080x 데이터들이 joint-corrected인지 질문했다. 2~3일간 관절 문제 가능성을 의심했다. 이 문서 작성 시점에는 모든 source episode에 대한 joint4 보정 적용 여부를 완전히 확정했다고 간주하지 않는다. 관련 데이터를 다시 조합할 경우 manifest/source 경로와 보정 산출물의 provenance를 먼저 확인한다.

## 다음 Codex가 우선 확인할 것

1. 현재 tmux session이 살아 있는지 `tmux ls`로 확인한다.
2. 로그는 전체를 읽지 말고 마지막 training progress/100-step 로그를 뽑아 현재 step, 속도, ETA를 계산한다.
3. `nvidia-smi`에서 학습 프로세스가 물리 GPU 1에만 있는지 확인한다.
4. 종료 후 `TRAIN_EXIT_STATUS=0`, 마지막 checkpoint 및 W&B summary를 확인한다.
5. 평가 전 CAN, 카메라 serial, park/align 상태를 확인한다.
6. 기존 출력/모델을 삭제하거나 덮어쓰지 않는다.

## 대화/협업 방식 메모

- 사용자는 장시간 무응답을 매우 답답해한다. 도구 실행이 길면 60초보다 훨씬 짧은 간격으로 현재 상태를 짧게 알려준다.
- 단순 진행 조회는 맨 마지막 진행 줄만 읽고 즉시 답한다.
- 명령은 복사 가능한 한 줄 또는 정확한 `\` 줄바꿈 형식으로 제공한다.
- 명령 안에서 경로 중간에 임의 줄바꿈을 넣으면 argparse가 별도 인자로 해석하므로 피한다.
- 실행/수정 전에 대상이 기존 자산인지 새 전용 자산인지 구분하고 명시한다.
