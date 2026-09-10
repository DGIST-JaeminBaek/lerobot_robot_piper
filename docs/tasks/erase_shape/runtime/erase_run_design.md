# 지우기 판정 설계

관련 코드: `scripts/tasks/erase_shape/runtime/erase_run.py`, `erase_check.py`,
`ink_metric.py`, `scripts/piper/inference/piper_infer_runner.py`

## 1. 전체 제어 흐름 (`erase_run.py main()`)

```
1. grab_judge_frame(top_cam)               # reference 프레임 촬영 (park 상태)
2. checker.set_reference(ref_frame)        # 도형 검출 + 초기 잉크량 기록
3. run_attempt_with_runner(...)            # InferenceRunner 스레드 시작 (아래 §2)
     - 정책이 30Hz로 제어 루프를 돈다 (--max-steps 2100 상한, release 시 조기 종료 — 아래 참고)
     - park_on_exit(기본 True)이 켜져 있고 안전 트립이 안 걸렸으면 종료 후 park (아래 참고)
4. judge_frame_from_video(record_path)     # 녹화 영상의 마지막 프레임을 그대로 씀
     - 실패 시(크롭본이라 좌표계 안 맞음 등) grab_judge_frame()으로 새로 촬영
5. checker.check(after_frame, target)      # 최종 판정
```

시도는 **1회로 고정**이고 재시도는 없다 — VLA가 release나 max-steps까지 스스로
제어하게 두고, 그 결과를 그대로 채점한다.

### `--stop-on-release`가 "release"를 확정하는 방식

한 프레임만 보고 즉시 끝내지 않는다. `ReleaseDetector`
([piper_infer_runner.py:2056-2119](../../../../scripts/piper/inference/piper_infer_runner.py#L2056-L2119))가
3단계로 확인한다:

1. **파지 확정** — 그리퍼 명령이 15프레임(0.5초) 연속 8~35 범위 안에 있고, 그 구간
   값이 3 이상 안 흔들려야("평평해야") "잡았다"로 본다. 지우개를 집으러 갈 때
   그리퍼를 여는 램프 동작이 이 범위를 스쳐 지나가는 걸 파지로 오인하지 않기
   위한 조건이다(실측 2026-08-18: 폭 조건이 없으면 집기도 전에 발화했다).
2. **최소 유지 시간** — 그 파지가 120프레임(4초) 이상 이어져야 한다. 못 미치면
   "헛놓기"로 보고 감지기를 리셋해 다시 잡을 기회를 준다(실측 근거: 시연의 진짜
   파지 지속은 최소 246프레임).
3. **release 확정** — 그리퍼가 파지 수준에서 15 이상 벌어진 상태가 5프레임 연속
   이어져야 "진짜 놓았다"로 본다. 1프레임짜리 노이즈로 실물 시도를 끊으면
   되돌릴 수 없어서다.

끊는 신호는 **그리퍼 놓기 하나뿐이다.** 한때 관절이 3초간 거의 안 움직이면
("정지") 같이 끊었지만 2026-08-21에 뺐다 — "정말 놓아서 끝난 것"과 "확신 없이
멈칫한 것"이 같은 종료 사유(`release`)로 섞여 판정이 흐려졌다. 지금은 멈칫해서
스스로 못 끝내는 시도는 `--max-steps` 상한에 맡기고, 종료 사유가 `release`와
`cutoff`로 깔끔히 갈리게 둔다.

### park은 조건부다

`park = settings.park_on_exit and not robot.safety_tripped`
([piper_infer_runner.py:1514](../../../../scripts/piper/inference/piper_infer_runner.py#L1514)).
release/cutoff 차이는 안 따지지만, 안전 트립이 걸리면 park하지 않는다.

**안전 트립**: 지우기 게이트와 무관한 CAN effort(토크) 과부하 컷오프. 매 제어
스텝마다 관절 실측 effort를 `safety_effort_limit`(기본 8.0 N·m)와 비교해 넘으면
`_trip_safety()`가 발동 — 트립 순간 자세를 계속 재전송(리더/정책 명령 무시)하고,
`safety_on_overload="park"`(기본값)면 별도 스레드로 자체 parking까지 한다
([piper_follower.py:354-357, 455-512](../../../../lerobot_robot_piper/piper_follower.py#L354-L512)).

안전 시스템이 이미 처리 중(또는 안 하기로 설정됨)이라 중복 파킹을 피하는 것이다.

## 2. 카메라·녹화

- 카메라: RealSense top(시리얼 고정, `--top-cam`) 필수, wrist(`--wrist-cam`) 선택.
  top 1280×720을 `--top-crop`(기본 `280,0,720`)으로 잘라 정책 입력으로 쓴다.
- 녹화(`--mode augment`)는 `piper_infer_runner.py`의 `RolloutRecorder`가 담당한다 —
  `LeRobotDataset.add_frame()`을 배경 스레드+큐로 비동기 호출해서 30Hz 제어 루프가
  디스크 쓰기 때문에 밀리지 않게 한다. `LeRobotDataset` 자체에도 카메라당 4스레드
  (teleop `lerobot-record`와 같은 값)를 넘겨 이미지 쓰기를 병렬화한다 — 예전엔
  스레드 0개(동기 쓰기)라 raw 1280×720 2카메라 기준 프레임당 80.5ms가 걸려 30Hz
  예산(33.3ms)을 못 따라가 큐가 넘쳐 드롭이 났다(2026-08-21 실측: 687스텝 중
  230프레임 드롭). 스레드를 늘린 뒤로는 드롭이 재현되지 않았다.
- 코덱은 `configs/recording.env`의 `VCODEC`(teleop 녹화와 같은 값)을 따른다 —
  안 정해져 있으면 `h264`(CPU). 이 프로젝트 설정은 `hevc_nvenc`(GPU)라 teleop과
  저장 형식을 통일했다. 실측으로는 CPU h264가 프레임당 더 빠르고(2026-08-21:
  CPU 848프레임 49.5ms/frame vs GPU 1187프레임 62.3ms/frame), 화질도 CPU의
  CRF(장면별 적응)가 GPU의 constant-QP(균일 압축, `lerobot`이 nvenc에 강제)보다
  이런 장면(정적 배경+움직이는 팔)엔 유리하지만, teleop 데이터셋과 형식을 맞추는
  쪽을 택했다. AV1(libsvtav1)만은 피할 것 — 이 환경 OpenCV가 못 읽어서 채점 GUI가
  세그폴트났었다(2026-08-18). h264/hevc는 둘 다 cv2로 바로 읽힌다.
- **파킹 중에도 녹화가 이어진다.** 예전엔 파킹 끝나고 한 장만 찍어 영상 끝에
  붙였는데, 그러면 "마지막 추론 프레임(팔이 도형 옆)"에서 "파킹 완료 프레임(팔이
  화면 밖)"으로 바로 점프해 영상이 튀었다(2026-08-19). 지금은 `robot.parking()`을
  별도 스레드로 돌리면서, 메인 스레드가 그 사이를 기존 제어 주기(fps)로
  `robot.get_observation()`을 계속 읽어 이어 찍는다(`piper_infer_runner.py`
  파킹 루프, `judge_frame_in_video` 플래그로 켜짐).

## 3. 캡처 타이밍 — 왜 이 두 순간인가

| 시점 | 방법 | 비고 |
|---|---|---|
| **reference** (시도 시작 **직전**) | `grab_judge_frame()`으로 top 카메라를 **직접** 새로 연다(RealSense 파이프라인 별도 오픈, warmup 3s, 1280×720 bgr8) | 정책 관측용 카메라 핸들과 분리 — 시도 사이엔 러너의 카메라 핸들이 없고, 판정기는 크롭 전 원본 좌표계(board ROI)가 필요해서 요구사항이 다르다 |
| **judge/after** (시도 종료, **파킹 완료 후**) | 파킹 이어찍기의 **마지막 원본 프레임**(압축 전, 러너가 `judge_frame_raw`로 들고 있다가 넘김). 없으면 녹화 영상의 마지막 프레임(`judge_frame_from_video`), 그것도 안 되면 `grab` | §2의 파킹 중 이어찍기 덕분에 "마지막 프레임 = 팔이 완전히 빠진 상태"가 보장된다. 새로 촬영(`grab`)을 기본으로 안 쓰는 이유는 워밍업 때문에 몇 초 늦어져 그 사이 사람이 보드를 정리해버리는 일이 반복됐기 때문(2026-08-20) |

**두 프레임 다 압축 전 원본이어야 한다.** 예전엔 judge만 녹화 영상에서 디코딩해서
썼는데, 영상은 crf/qp=30으로 인코딩된 뒤라 무손실인 reference와 화질이 비대칭이었다.
실측(2026-08-21): 같은 사진을 h264 crf30에 통과시키면 선명도가 절반으로 떨어지고
(Laplacian var 118→60) 잉크 비율이 6.4% 줄어든다 — `ink_after`만 과소평가되므로
`erased_frac`이 성공 쪽으로 계통 편향된다(90% 임계 근처에서 +0.6~2%p).

이 두 순간에만 재는 이유: 팔이 도형을 가리지 않는 유일한 구간이라서다(지우는
동안엔 팔이 항상 도형 위에 있다 — 중간 판정 불가).

`reference_frame.png`/`judge_frame.png`로 롤아웃 폴더에도 저장된다 — 오프라인
재채점(`erase_eval.py`)이 라이브 게이트와 **같은 사진**을 보게 하기 위해서다(다른
사진을 쓰면 같은 최종 상태를 보고도 서로 다른 숫자가 나온 적이 있다, 2026-08-18:
77.8% vs 86.4%).

## 4. 도형 검출 (`ink_metric.detect_shapes`)

```
1. board bbox로 crop → grayscale, HSV 채도 채널 계산
2. white = grayscale 상위 90 percentile (보드 흰 영역 밝기)
3. ink 마스크 = (gray < white × dark_ratio)
4. morphologyEx(CLOSE, 9×9) — 손그림 획 끊김을 닫는다
5. findContours(RETR_EXTERNAL)
6. 배제 구역(exclude, 기본은 보드에 눌어붙은 테이프 자국 좌표)과
   EXCLUDE_OVERLAP(0.5) 이상 겹치는 컨투어는 통째로 버림
   (마스크를 지우지 않고 컨투어 자체를 버려야 한다 — 마스크를 지우면
   겹친 물체가 반으로 쪼개져 성질이 바뀐 조각이 필터를 통과해버린다)
7. 채도 필터(MAX_SAT=25, contour bbox 평균 채도) — 나무 지우개 블록(평균 채도 84)
   제거. **병합 전에** 적용해야 한다(나중에 하면 블록이 옆 도형과 한 덩어리로 묶임)
8. 팔 필터(BORDER_MARGIN=3) — 컨투어가 보드 오른쪽 경계에 닿아 있으면 로봇 팔로
   본다(팔은 항상 그 쪽 몸체와 이어져 있어 경계에 닿는다). 손그림 도형은 보드
   안에 갇혀 있어 안 닿는다
9. merge_nearby(gap=30px) — 가까운 조각을 한 도형으로 병합(도형 간 최소 간격은
   100px 이상이라 안전)
10. 병합 결과가 다시 경계에 닿으면 한 번 더 팔 필터(놓친 팔 조각이 섞였을 수 있음)
11. 크기/종횡비 필터(MIN_SHAPE_AREA=2000, MAX_ASPECT=4.0, MAX_SPAN=0.7×보드변)
12. classify(contour) — **bbox 병합 후에** 분류해야 한다(병합 전에 하면 끊긴 획을
    가진 삼각형이 원으로 오분류된다). convex hull 면적/최소외접사각형 면적(extent)
    으로 판정: <0.63 삼각형, >0.88 사각형, 그 사이(원 이론값 0.785와 겹치는 구간)는
    approxPolyDP 꼭짓점 수로 재분류(≤4 사각형, 그 외 원)
13. bbox에 PAD(12px) 여유를 붙이고 화면 위→아래 순으로 정렬
```

같은 종류 도형이 여러 개 검출되면 `label#i`(예: `circle#0`, `circle#1`)로 키를
분리한다 — 안 그러면 시계열이 이어붙어 통계가 깨진다.

## 5. 판정 공식

target 도형 하나에 대해:

```
ink(frame, bbox) = bbox 안에서 (gray < ink_thr) 인 픽셀의 비율
ink_thr           = white × dark_ratio          (§4의 white, dark_ratio=0.72)

erased_frac = (ink_before − ink_after) / ink_before      [0, 1]로 클립
```

`ink_before`/`ink_after`는 §3의 reference/judge 프레임에서만 잰다. 성공 =
`erased_frac ≥ 0.90`.

**종료 사유**(`termination`)는 성공 판정과 분리해서 따로 기록한다 — `release`(그리퍼를
스스로 놓음) / `cutoff`(`--max-steps`까지 다 씀). "로봇이 끝났다고 놓은 것"과 "실제로
다 지운 것"은 다른 사건이라서다.

**실패 유형은 자동으로 추측하지 않는다.** 채점기는 `erased_frac`과 `termination`만
내고, 실패했을 때 왜 실패했는지(파지 실패/접촉 실패/미완료 등)는 사람이 GUI에서
직접 고른다.

## 6. 벤치마크 조건 — 도형을 어디에 어떤 크기로 그리나

`erased_frac`은 시도 사이에 **보드 상태가 같아야** 비교가 된다. 도형이 크면 지울 면적이
넓어 같은 시간 안에 못 끝내고, 자리가 다르면 학습 분포 안/밖이 갈린다. 그래서 매 시도
전에 사람이 도형을 그릴 때 **위치와 크기를 315 기준에 맞춘다.**

**기준은 315다.** `candidate_assets/records_105`의 315개(circle/triangle/rectangle 각
105개) 원본 녹화 첫 프레임에서 도형 bbox를 측정해 만들었다. 첫 프레임을 쓰는 이유는
그 시점에 팔이 홈 자세라 보드를 안 가리기 때문이다.

**6개 zone(위치)**: `cy=300`으로 상/하 2행을 나누고(y가 뚜렷한 이봉분포 — 위 74~200 /
아래 410~590에 몰리고 중간대는 315개 중 1개뿐), 각 행에서 `cx`를 k-means(k=3)로 좌/중/우
3열로 갈라 6개 클러스터의 median을 대표 위치로 쓴다.

**bounding box(크기)**: zone별 박스의 (w, h)는 그 자리에 실제로 그려졌던 도형들의
median이다. 도형 종류마다 계통적으로 다르다(315 전체 median, `--size-source global`):

| 도형 | w × h | 특징 |
|---|---|---|
| circle | 127 × 124 | 가로로 가장 넓다 |
| triangle | 106 × 129 | 세로로 길고 가로가 좁다 |
| rectangle | 118 × 119 | 거의 정사각 |

zone별로 쪼갠 값(`--size-source per-zone`)도 있지만 조합당 표본이 10~22개까지 줄어
세션 편차에 median이 흔들린다(실측: 315의 top-left/circle은 0811 세션이 작게, 0805가
크게 그려 대표성이 떨어졌다). 그래서 평가 GUI는 표본이 도형당 105개인 **global**을 쓴다.

**현장 사용**: `block_alignment_tool.py --shape-zones 315 --target-shape <도형>
--size-source global`이 이 박스를 카메라 영상 위에 겹쳐 그린다(circle만 안쪽에 타원
가이드를 같이 그리고, triangle/rectangle은 박스만). 그려 넣은 뒤 `c`를 누르면 판정과
같은 파이프라인(`detect_shapes`/`classify`)으로 검사해서, target 종류로 분류되고 zone
중심에서 70px 안이면 `PASS`가 뜬다. 즉 이 검사를 통과하면 실제 판정도 통과한다.

> **132 체크포인트를 평가할 때 주의**: `pick_up_the_eraser_0727_0812_0813am_132`
> 학습셋에는 315의 **좌측 열(cx≈325~360)에 해당하는 도형이 하나도 없다**. 그 자리에
> 그리면 학습 분포 밖에서 평가하는 셈이다. 이 학습셋 기준 위치는 상/하 × 좌/우
> **4개**뿐이고 `--shape-zones 132`로 볼 수 있다(2026-08-22 실측, 132/132 전부 측정).
> 비교 그림: `outputs/analysis/erase_shape/zone_compare_132_vs_315/`.

## 7. 알아둘 리스크

**노출 드리프트**: `EraseChecker.set_reference()`가 기준 프레임에서 `ink_thr`을 한
번만 계산해 재사용한다. 시도 전/후 사이 RealSense 자동 노출이 움직이면
`erased_frac`이 조명 변화를 잉크 변화로 오독한다 — 의심되면 시도 전/후 프레임의
보드 흰 영역 밝기를 비교하고, 필요하면 노출·화이트밸런스를 수동 고정한다.

**park 자세**: `follower.parking()`이 보드를 가리거나 도형에 그림자를 드리우면
판정이 무의미해진다 — 새 환경에서는 눈으로 확인할 것.

## 관련 문서

- 실행 명령·환경 세팅: [eval_session_howto.md](../evaluation/eval_session_howto.md)
- HIL(사람 개입) 설계 — 지금 안 씀: [hil_intervention_design.md](hil_intervention_design.md)
