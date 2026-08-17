#!/usr/bin/env python3
"""판정용 카메라의 노출 드리프트를 잰다 (설계 문서 §4.5 / §7.2-2단계).

왜 최우선인가 — `EraseChecker.set_reference()`는 기준 프레임에서
`ink_thr = white * dark_ratio`를 **한 번** 계산하고 `check()`가 그 값을 재사용한다.
녹화 데이터에서는 같은 영상 안이라 문제가 없었지만, 추론 중에는 시도 전과 후 사이에
수십 초가 흐른다. 그동안 RealSense 자동 노출이 움직이면:

    노출이 밝아지면 → 잉크가 덜 검게 잡힘 → erased_frac이 부풀어 오름 (거짓 성공)
    노출이 어두워지면 → 반대 (거짓 실패)

즉 **조명 변화를 잉크 변화로 오독한다.** 이게 크면 여기서 나온 성공률은 전부
의미가 없으므로, 실험 데이터를 쌓기 전에 확인해야 한다.

로봇을 건드리지 않는다 — 카메라만 읽는다.

    python check_exposure_drift.py --duration 90 --interval 5

보는 값:
  white   보드 흰 영역 밝기(90 퍼센타일). ink_thr = white * 0.72의 근거.
  ink     기준 프레임의 ink_thr로 잰 잉크 픽셀 비율. **보드를 안 건드렸으면
          이 값이 안 변해야 한다.** 변한 만큼이 그대로 판정 오차다.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ink_metric as IM  # noqa: E402
from erase_run import grab_judge_frame  # noqa: E402

# 이 이상 흔들리면 판정을 못 믿는다. 근거: distractor 노이즈 플로어가 p95 0.037이고
# 성공 임계 여유(target 0.987 vs 임계 0.9)가 0.087이다. 노출만으로 erased_frac이
# 0.03 넘게 움직이면 노이즈 플로어와 같은 크기가 되어 분리도가 무너진다.
ERASED_TOLERANCE = 0.03


def measure(frame, board, dark_ratio, ink_thr=None):
    bx, by, bw, bh = board
    roi = cv2.cvtColor(frame[by : by + bh, bx : bx + bw], cv2.COLOR_BGR2GRAY)
    white = float(np.percentile(roi, 90))
    thr = ink_thr if ink_thr is not None else white * dark_ratio
    hsv_sat = float(cv2.cvtColor(frame[by : by + bh, bx : bx + bw],
                                cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    return {
        "sat": hsv_sat,
        "white": white,
        "ink_thr_used": thr,
        # 기준 임계로 잰 잉크 비율 — 판정이 실제로 쓰는 값과 같은 계산
        "ink": float((roi < thr).mean()),
        "mean": float(roi.mean()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", default="327122074262", help="top 카메라 시리얼")
    p.add_argument("--duration", type=float, default=90.0, help="총 관측 시간(초)")
    p.add_argument("--interval", type=float, default=5.0, help="샘플 간격(초)")
    p.add_argument("--board", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=IM.DEFAULT_BOARD)
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--warmup", type=float, default=3.0,
                   help="샘플당 카메라 워밍업(초). erase_run의 판정 경로와 같아야 "
                        "의미가 있다. 3.0 미만이면 auto-WB가 안 잡힌다")
    p.add_argument("--save-frames", type=Path, default=None,
                   help="첫/마지막 프레임을 저장할 디렉터리")
    p.add_argument("--plot", type=Path, default=None)
    a = p.parse_args(argv)

    print(f"[INFO] {a.duration:.0f}초 동안 {a.interval:.0f}초 간격으로 잰다. "
          f"보드를 건드리지 말 것 — 변화가 곧 오차다.")

    samples = []
    first_frame = None
    ref_thr = None
    t_end = time.time() + a.duration
    while time.time() < t_end:
        t = time.time()
        # ★ warmup은 erase_run이 실제 판정에 쓰는 값과 같아야 한다(기본 3.0s).
        #   여기를 0.5s로 줄였다가 크게 헤맸다 — 매 샘플이 파이프라인을 새로 열기
        #   때문에 auto-WB가 수렴하기 전 프레임을 재게 되고, 보드 채도가 12~15가
        #   아니라 129로 나온다. 그 프레임에서는 도형이 MAX_SAT 필터에 걸려
        #   통째로 미검출되고, 마치 카메라가 고장난 것처럼 보인다.
        frame = grab_judge_frame(a.serial, warmup_s=a.warmup)
        if first_frame is None:
            first_frame = frame.copy()
            ref_thr = measure(frame, a.board, a.dark_ratio)["ink_thr_used"]
        m = measure(frame, a.board, a.dark_ratio, ink_thr=ref_thr)
        m["t"] = t
        samples.append(m)
        print(f"  +{len(samples)*a.interval:5.0f}s  white={m['white']:6.2f}  "
              f"ink={m['ink']:.5f}  mean={m['mean']:6.2f}  채도={m['sat']:6.1f}")
        sleep = a.interval - (time.time() - t)
        if sleep > 0:
            time.sleep(sleep)
        last_frame = frame

    if len(samples) < 2:
        print("샘플이 부족하다.", file=sys.stderr)
        return 1

    whites = np.array([s["white"] for s in samples])
    sats = np.array([s["sat"] for s in samples])
    inks = np.array([s["ink"] for s in samples])

    # ★ 핵심 수치. 잉크량이 흔들린 폭을 '기준 대비 erased_frac 오차'로 환산한다.
    #   판정식이 (before - after)/before 이므로, before를 첫 샘플로 놓으면
    #   각 샘플의 겉보기 erased가 (ink[0] - ink[i]) / ink[0] 이다.
    base = inks[0]
    apparent = (base - inks) / base if base > 0 else np.zeros_like(inks)

    print("\n── 결과 " + "─" * 50)
    print(f"  white 밝기   {whites.min():.2f} ~ {whites.max():.2f}  "
          f"(변동 {100*(whites.max()-whites.min())/whites.mean():.2f}%)")
    print(f"  잉크 비율    {inks.min():.5f} ~ {inks.max():.5f}")
    # 학습 데이터(0802/0805) 보드 채도는 12~13이다. 여기서 크게 벗어나면 화이트
    # 밸런스가 안 잡힌 것이고, 판정뿐 아니라 정책 입력도 분포 밖이 된다.
    print(f"  보드 채도    {sats.min():.1f} ~ {sats.max():.1f}  "
          f"(학습 데이터 12~13 / MAX_SAT={IM.MAX_SAT})")
    print(f"  ★ 노출만으로 생긴 겉보기 erased_frac: "
          f"{apparent.min():+.4f} ~ {apparent.max():+.4f}  "
          f"(절대 최대 {np.abs(apparent).max():.4f})")

    worst = float(np.abs(apparent).max())
    ok = worst <= ERASED_TOLERANCE
    print(f"\n  허용치 {ERASED_TOLERANCE} 대비: "
          f"{'통과 — 판정을 믿어도 된다' if ok else '초과 ★ 노출·화이트밸런스 수동 고정 필요'}")
    if not ok:
        print("    → RealSense의 auto_exposure / white_balance를 끄고 고정값을 주거나,")
        print("      매 프레임 white를 재추정하도록 EraseChecker를 고칠 것.")

    if a.save_frames:
        a.save_frames.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.save_frames / "first.png"), first_frame)
        cv2.imwrite(str(a.save_frames / "last.png"), last_frame)
        print(f"  프레임 저장: {a.save_frames}")

    if a.plot:
        from plot_ko import plt, use_korean
        use_korean()
        ts = (np.array([s["t"] for s in samples]) - samples[0]["t"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(ts, whites, "o-", color="#c78d00")
        ax1.set_ylabel("보드 흰 영역 밝기")
        ax1.set_title("판정용 카메라 노출 드리프트 (보드 정지 상태)")
        ax2.plot(ts, apparent, "o-", color="crimson")
        ax2.axhspan(-ERASED_TOLERANCE, ERASED_TOLERANCE, color="green", alpha=0.12,
                    label=f"허용치 ±{ERASED_TOLERANCE}")
        ax2.axhline(0, color="0.5", lw=0.8)
        ax2.set_ylabel("겉보기 erased_frac")
        ax2.set_xlabel("경과 시간 (초)")
        ax2.legend()
        for ax in (ax1, ax2):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print(f"  그림 저장: {a.plot}")

    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
