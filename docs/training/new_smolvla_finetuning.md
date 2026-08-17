# SmolVLA 재학습 (2026-08-08~) — 진행 기록

## 1. 학습하려는 4가지 전략

`erase the shape` task 대응 여부를 보려고 SmolVLA를 4가지 설정으로 학습시키기로 했다.

1. 기본 (top camera + wrist camera)
2. top view만
3. top camera + wrist camera + memory module
4. top camera + memory module

데이터는 0728을 제외한 0727/0802/0804/0805에서 circle/triangle/rectangle 각 50개씩 골라 총 150 episode로 구성했고, task는 `erase the shape` 하나로 통합했다(방해 도형이 없는 시나리오라 도형별로 나눌 필요가 없다고 판단). GOP=1(소프트웨어 hevc)로 빌드하고 전수 디코딩 검증까지 통과시켰다.

```
records/outputs/erase_the_shape_150            (top+wrist, 653MB)
records/outputs/erase_the_shape_150_top_only   (top only, 366MB)
```

memory module(전략 3, 4)은 별도 경로에서 준비 중이라 아직 전달되지 않았다. 그래서 이번엔 memory module이 없는 전략 1, 2부터 먼저 진행하기로 했다.

## 2. 전략 1·2를 GPU 0/1에 각각 올려서 동시 진행 시도

GPU가 2장(RTX 5090 32GB × 2) 있어서, 전략 1(top+wrist)은 GPU 0에, 전략 2(top-only)는 GPU 1에 올려 동시에 학습시키기로 했다. 두 run 다 기존 `smolvla_erase_shape_512` 레시피와 동일한 하이퍼파라미터(batch_size=8, steps=30000, freeze_vision_encoder=true, train_expert_only=true, BF16 등)를 쓰고, 이번엔 W&B 로깅을 추가로 켰다.

## 3. GPU 0(전략 1)은 온도 때문에 중단, 지금은 GPU 1(전략 2)만 진행 중

학습 중 GPU 0 온도가 85°C까지 올라가서(RTX 5090 쓰로틀 시작점은 90°C라 위험 수준은 아니었지만) 전략 1 프로세스를 SIGINT로 중단시켰다. 로컬에 저장된 체크포인트는 5000 step까지이고(`save_freq=5000`), 30000 step은 못 채웠다. 이 기록은 `outputs/train/smolvla_erase_shape_150_s1_topwrist_aborted_thermal85c`로 폴더명을 바꿔 보존했다.

같은 `--wandb.run_id`로 무심코 재시작했다가 기존 W&B run(이름/태그/`train/loss` 기록)이 섞여서 오염됐다. 로컬엔 체크포인트가 저장되기 전에 죽여서 피해가 없었지만, W&B `s1_topwrist` run은 복구 불가 판단하에 삭제했다. **다음부터 재시도할 땐 `--wandb.run_id`를 반드시 새 이름으로 바꿀 것.**

GPU 1(전략 2, top-only)은 계속 진행 중이다. 참고로 GPU 1은 모니터가 연결된 카드인데도 GPU 0보다 더 낮은 온도(60도대)를 유지했다 — 디스플레이 부담보다는 (a) 전략 1이 카메라 2개를 처리해 연산량 자체가 더 크고 (b) GPU 0의 물리적 슬롯 위치(PCI `01:00.0`, GPU 1은 `03:00.0`)가 통풍이 더 안 좋을 가능성이 있다고 보고 있다.

## 4. 속도는 확실히 빨라졌다

체크포인트 저장 시각으로 실측한 GPU 1(전략 2, top-only, GOP=1)의 속도:

```
checkpoints/005000  저장 15:07
checkpoints/010000  저장 15:13
→ 5000 step / 6분 ≈ 13.9 step/s
```

기존 GOP=250(정정: 실제로는 ~200, 아래 5절 참고) 학습(`smolvla_erase_shape_512`, 2.07 step/s) 대비 체크포인트 저장 시각 기준 약 6.7배로 보였으나, 이 측정 방식(체크포인트 파일 저장 시각 차이)은 저장 I/O 시간이 섞여 부정확했다. 정확한 수치는 5절 참고.

## 5. 진짜 병목은 GOP 설정이었다 — 그리고 NVENC이 GOP 옵션을 무시하는 버그를 발견했다

체크포인트 저장 시각 대신 W&B에 매 100 step 기록되는 `train/update_s`(GPU 연산 시간)와 `train/dataloading_s`(데이터 로딩 시간)를 직접 비교해보니 더 정확한 그림이 나왔다.

```
                  update_s(GPU 연산)  dataloading_s   합(1step)   이론 step/s
top-only(전략2)      0.0758s           0.0020s        0.0778s     12.85
top+wrist(전략1)     0.0956s           0.0028s        0.0985s     10.15
```

top+wrist가 top-only보다 스텝당 더 오래 걸리는 게 정상이다(카메라 2배 처리). 체크포인트 시각 기반으로 계산했던 "top+wrist가 top-only보다 빠르다"는 이전 관찰은 측정 오차였다.

**`smolvla_erase_shape_512`의 실제 GOP을 직접 열어서 세어봤다:**

```
30,545프레임 중 키프레임 153개, 평균 간격 199.6프레임
```

문서에 GOP=250이라고 적혀 있었지만(스크립트 기본값), 실제로는 250이 아니었다. 원인은 이번 세션에서 새로 발견한 버그다 — **lerobot의 `_get_codec_options()`(`lerobot/datasets/video_utils.py`)가 `HW_ENCODERS`(NVENC/VAAPI/QSV 등, VideoToolbox 제외)에는 GOP 옵션 자체를 안 넘긴다.** `erase_the_shape_512`는 `--vcodec hevc_nvenc`로 빌드됐기 때문에, `--gop-size`로 뭘 지정하든(심지어 기본값 250조차) 인코더에 전달되지 않고 NVENC 자체 내부 기본값(~200)으로 만들어진 것이다. 오늘 150-episode 데이터셋을 GOP=1로 재빌드하려다가 똑같은 문제로 두 번 빌드해야 했던 것과 동일한 원인이다.

**정정된 비교**(`train/update_s`+`dataloading_s` 기준, GOP≈200 vs GOP=1, 둘 다 top+wrist):

```
GOP≈200 (smolvla_erase_shape_512, hevc_nvenc)   2.07 step/s
GOP=1   (지금, 소프트웨어 hevc)                 ≈10.15 step/s
→ 약 4.9배
```

즉 학습 속도 병목의 핵심은 카메라 개수도, 데이터셋 크기(60→150)도 아니라 **GOP 설정(정확히는 GOP=1이 아니면 생기는 디코딩 되감기 비용)**이었다. 그리고 그 GOP 설정은 소프트웨어 인코더(`hevc`)를 써야만 의도대로 먹힌다 — NVENC 계열은 GOP 옵션을 조용히 무시한다는 게 이번에 새로 확인된 사실이다.

## 6. memory module 없는 전략 1·2 학습 완료

`s1_topwrist`를 GPU 0에서 재시도(`--wandb.run_id=s1_topwrist_retry`)해서 30,000 step까지 정상 완료했다. 전략 2(top-only)는 이미 앞서 완료돼 있었으므로, 이제 **memory module이 없는 두 전략(1, 2) 다 학습이 끝났다.**

```
outputs/train/smolvla_erase_shape_150_s1_topwrist   step 30000  (top+wrist)
outputs/train/smolvla_erase_shape_150_s2_toponly    step 30000  (top only)
```

둘 다 체크포인트 6개(5000 간격) + `last` 심볼릭 링크까지 정상 저장됐고, W&B에도 `s1_topwrist_retry`/`s2_toponly` run으로 loss·시스템 지표가 남아있다.

## 다음 계획

memory module(전략 3, 4)이 준비되는 대로 같은 150-episode 데이터(top+wrist / top-only)에 적용해서 나머지 두 전략을 마저 학습시킬 예정이다. GPU 0 슬롯이 원래 더 뜨거운 경향이 있는지는 이번 재시도에서도 계속 관찰했고, 필요하면 GPU 0/1 워크로드를 바꿔서(예: top-only를 GPU 0에서, top+wrist를 GPU 1에서) 온도 차이가 슬롯을 따라가는지 워크로드를 따라가는지 교차 확인하는 것도 고려 중이다.
