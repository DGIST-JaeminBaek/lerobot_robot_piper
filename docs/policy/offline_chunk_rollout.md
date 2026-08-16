# Offline Action Chunk Rollout

## 1. 목적

학습된 SmolVLA가 생성하는 action chunk 궤적을 실제 Piper 없이 확인한다. 학습
데이터 episode의 observation을 일정 frame 간격으로 다시 입력하고, 매 입력에서
예측한 action chunk를 이어 붙여 전체 episode 길이의 예측 궤적을 만든다.

```text
Dataset의 첫 observation 입력
  → SmolVLA action chunk 예측
  → 앞 N개 action을 예측 궤적에 추가
  → Dataset도 N frame 앞으로 이동
  → 해당 frame의 observation을 다시 입력
  → Episode 끝까지 반복
```

Dataset과 policy의 FPS가 같고 action 하나를 frame 하나에 실행한다고 가정한다.
따라서 별도의 frame 수 예측 모델을 사용하지 않는다.

```text
50 action 사용 → dataset 50 frame 이동
20 action 사용 → dataset 20 frame 이동
10 action 사용 → dataset 10 frame 이동
```

## 2. 검증의 성격

이 방식은 물리 시뮬레이션이나 완전한 closed-loop rollout이 아니다. 다음
observation은 예측 action의 결과로 생성되지 않고 녹화된 전문가 trajectory에서
가져온다.

```text
Policy가 예측한 action
  → RViz에서 관절 궤적으로만 사용

다음 이미지와 state
  → Train dataset의 미래 frame에서 가져옴
```

따라서 다음 항목을 확인하는 teacher-forced offline 검증으로 해석한다.

- 예측 관절 궤적이 dataset action의 전체 흐름을 따라가는지
- 지우개를 집는 gripper 변화가 나타나는지
- Chunk 경계에서 action이 크게 튀는지
- Joint별 예측 오차가 어느 구간에서 커지는지
- Joint 및 gripper 정규화 범위를 벗어나는지

지우개 접촉, 파지 성공, 물체 이동 및 실제 지우기 결과는 검증하지 않는다.

## 3. 구현 파일

### Rollout 생성

`scripts/tools/piper_offline_chunk_rollout.py`

- LeRobotDataset에서 지정 episode를 로드한다.
- Dataset과 함께 저장된 statistics로 policy pre/postprocessor를 구성한다.
- 각 decision frame의 state, top image 및 wrist image를 SmolVLA에 입력한다.
- `predict_action_chunk()`로 전체 chunk를 얻는다.
- `actions_per_step`개를 예측 궤적에 붙이고 dataset cursor도 같은 수만큼 이동한다.
- 같은 frame 구간의 dataset action과 비교한다.
- NPZ, JSON 및 비교 그래프를 저장한다.

Piper, CAN 및 `robot.send_action()`은 사용하지 않는다.

### RViz 재생

`scripts/tools/piper_offline_rollout_rviz.py`

- 저장된 NPZ에서 예측 또는 dataset 정답 action을 읽는다.
- 정규화된 Piper action을 URDF joint 단위로 변환한다.
- ROS2 `/joint_states`에 publish한다.
- 실제 로봇이나 CAN에는 연결하지 않는다.

## 4. Rollout 실행

학습에 사용한 512×512 데이터셋과 15,000-step 체크포인트의 예:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/tools/piper_offline_chunk_rollout.py \
  --dataset-root records/0727/erase_the_shape_512 \
  --episode 0 \
  --policy-path outputs/train/smolvla_erase_shape_512/checkpoints/015000/pretrained_model \
  --task "erase the shape" \
  --device cuda \
  --actions-per-step 50
```

`CUDA_VISIBLE_DEVICES=1`은 두 번째 GPU를 사용하려는 경우에만 지정한다.
`--actions-per-step`은 policy의 `chunk_size=50`을 초과할 수 없다.

실행할 때 실제 robot 관련 설정이나 action offset 설정은 사용되지 않는다.

## 5. 저장 결과

기본 출력 경로:

```text
outputs/offline_rollout/<dataset>_ep<episode>_<checkpoint>/
```

파일 구성:

| 파일 | 내용 |
|---|---|
| `rollout_actions.npz` | 예측 action, dataset action, decision frame 및 각 action의 observation 출처 |
| `summary.json` | 전체·joint별 오차, chunk별 추론 시간과 경계 점프 |
| `trajectory_comparison.png` | 예측 action과 dataset action의 joint별 비교 그래프 |

NPZ의 주요 배열:

| Key | Shape | 내용 |
|---|---|---|
| `predicted_actions` | `(frames, 7)` | 이어 붙인 SmolVLA 예측 action |
| `expert_actions` | `(frames, 7)` | 같은 구간의 dataset action |
| `observation_source_frames` | `(frames,)` | 각 예측 action chunk를 만든 observation frame |
| `decision_frames` | `(chunks,)` | SmolVLA를 다시 호출한 dataset frame |

## 6. 15,000-step Episode 0 실행 결과

사용 조건:

```text
Dataset: records/0727/erase_the_shape_512
Episode: 0
Frames: 308
FPS: 30
Task: erase the shape
Checkpoint: 015000
Policy chunk size: 50
Actions per step: 50
Decision frames: 0, 50, 100, 150, 200, 250, 300
```

결과:

| 항목 | 값 |
|---|---:|
| 전체 MAE | 2.3898 |
| 전체 RMSE | 3.4895 |
| 최대 절대 오차 | 14.5404 |
| 정규화 범위 초과 | 0 |
| 최대 chunk 경계 점프 | 약 11.31 |

저장 위치:

```text
outputs/offline_rollout/erase_the_shape_512_ep0000_015000/
```

예측 궤적은 gripper 변화와 전체 관절 흐름을 대체로 따라갔다. 다만 50-frame
chunk 경계에서 눈에 띄는 불연속이 있으므로 이 결과만으로 실물 실행이 안전하다고
판정할 수 없다.

이 결과는 최종 학습 모델이 아니라 15,000-step 중간 체크포인트로 생성했다.
30,000-step 학습은 완료됐지만 같은 episode와 seed의 full rollout 재실행은 아직
하지 않았다.

## 7. 저장된 궤적의 RViz 재생

이 절차는 rollout 생성과 분리되어 있으므로 policy를 다시 로드하거나 추론하지
않는다. 먼저 [RViz 설정 문서](../rviz_setup.md)에 따라 RViz와
`robot_state_publisher`를 실행한다.

예측 궤적:

```bash
python scripts/tools/piper_offline_rollout_rviz.py \
  --rollout outputs/offline_rollout/erase_the_shape_512_ep0000_015000/rollout_actions.npz \
  --trajectory predicted \
  --fps 30
```

Dataset 정답 궤적:

```bash
python scripts/tools/piper_offline_rollout_rviz.py \
  --rollout outputs/offline_rollout/erase_the_shape_512_ep0000_015000/rollout_actions.npz \
  --trajectory expert \
  --fps 30
```

15,000-step 결과는 수치와 그래프까지 확인했으며 RViz player는 준비만 했다. 사용자
요청에 따라 RViz는 아직 실행하지 않았다.

## 8. 전체 episode 첫 chunk FK 분석

`scripts/tools/piper_first_chunk_fk_analysis.py`는 60개 episode의 첫 observation을
각각 입력해 첫 50-action chunk를 생성하고 Piper SDK `CalFK`로 end-effector
궤적을 계산한다.

목적:

- 모든 episode에서 지우개를 향한 초기 이동이 비슷하게 생성되는지 비교
- Joint target뿐 아니라 3D end-effector 경로와 endpoint 비교
- Circle 20개, triangle 20개, rectangle 20개 그룹 분포 비교
- 같은 RNG seed를 episode마다 복원해 sampling noise 영향을 줄임
- Global joint 범위 및 preview용 `max_relative_target` 적용 결과 비교

최종 30,000-step checkpoint로 60개 episode를 모두 실행했다.

```text
Policy: outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model
Episodes: 0~59
Chunk size: 50
Seed: 1000
Same noise per episode: true
Mean inference time: 0.1159초
```

재현 명령:

```bash
python scripts/tools/piper_first_chunk_fk_analysis.py \
  --dataset-root records/0727/erase_the_shape_512 \
  --policy-path outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model \
  --task "erase the shape" \
  --episodes all \
  --chunk-size 50 \
  --seed 1000 \
  --max-relative-target 5.0
```

출력 위치:

```text
outputs/first_chunk_fk/erase_the_shape_512_030000/
```

주요 파일:

| 파일 | 내용 |
|---|---|
| `first_chunk_fk.npz` | Episode별 raw/global/safe action과 FK 결과 |
| `first_chunk_fk.csv` | Episode·group·action step 단위 값 |
| `summary.json` | 추론 시간, clamp 수, EEF 분포 요약 |
| `joint_targets_global.png` | Joint별 첫 chunk 비교 |
| `eef_3d_absolute_global.png` | 절대 좌표 3D EEF 궤적 |
| `eef_3d_relative_global.png` | Episode 시작점 기준 상대 3D 궤적 |
| `eef_endpoints_relative_global.png` | 상대 endpoint 분포 |

확인된 요약값:

| 항목 | 값 |
|---|---:|
| 평균 추론 시간 | 0.1159초 |
| Global clamp 값 수 | 1,395 |
| Relative clamp 값 수 | 17 |
| Action step별 평균 XYZ spread | 27.24 mm |
| 최대 XYZ spread | 53.44 mm |

이 분석은 초기 궤적의 일관성을 보는 도구다. 몇 번째 action에서 실제로 지우개를
집는지, 접촉이 성공하는지, 물체를 지우는지는 FK만으로 판단할 수 없다. 모든 입력이
학습 데이터의 첫 frame이라는 점도 함께 고려한다.

## 9. 인간 승인형 Dataset/RViz preview

`scripts/tools/piper_human_approved_inference.py`의 안전 기본 모드로 dataset
observation을 입력하고, 생성된 chunk와 policy 입력 TOP/WRIST 이미지를 표시한 뒤
터미널에서 승인하는 흐름을 확인했다.

확인한 episode:

```text
0, 2, 36, 55
```

결과는 다음 경로에 저장되어 있다.

```text
outputs/human_approved_preview/erase_the_shape_512_ep0000/
outputs/human_approved_preview/erase_the_shape_512_ep0002/
outputs/human_approved_preview/erase_the_shape_512_ep0036/
outputs/human_approved_preview/erase_the_shape_512_ep0055/
```

이 결과는 이전 50-action 단위 RViz-only 실행에서 생성된 기록이다. 현재 실행기는
한 chunk를 기본 10-action 구간으로 나누어 각 구간을 별도로 preview하고 승인하도록
확장되었다. 실물 실행 경로의 상세 내용은
[인간 승인형 Policy 실행](human_approved_policy_execution.md)을 참고한다.

## 10. 결과 해석과 다음 검증

Train episode 결과는 policy가 학습 데이터를 외운 경우에도 좋게 나올 수 있다.
따라서 다음 순서로 확장한다.

1. 최종 30,000-step checkpoint로 episode 0 full rollout 재실행
2. `actions_per_step=50`, `20`, `10`의 chunk 경계 오차 비교
3. Circle, triangle 및 rectangle episode에서 각각 full rollout 반복
4. 학습에 포함하지 않은 validation episode로 동일 검사
5. Offline 결과를 확인한 뒤 인간 승인형 실물 실행 검토

## 11. 관련 문서

- [Policy 실행](README.md)
- [인간 승인형 Policy 실행](human_approved_policy_execution.md)
- [SmolVLA Finetuning](../training/smolvla_finetuning.md)
- [ROS 2/RViz/URDF 설정](../rviz_setup.md)
