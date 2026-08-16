#!/usr/bin/env python3
"""녹화된 데이터셋에 effort가 제대로 들어갔는지 검증.

사용법:  python check_effort.py <데이터셋_경로>

차원만 보면 SDK가 0을 돌려줘도 통과하므로, 실제 값 분포까지 확인한다.
"""
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

OK, BAD, WARN = "\033[92m✅", "\033[91m❌", "\033[93m⚠️ "
END = "\033[0m"


def main(root_str: str) -> int:
    root = pathlib.Path(root_str)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"{BAD} meta/info.json 없음: {info_path}{END}")
        return 1

    info = json.loads(info_path.read_text())
    st = info["features"].get("observation.state")
    if st is None:
        print(f"{BAD} observation.state 피처가 없음{END}")
        return 1

    names = st["names"]
    eff_i = [i for i, n in enumerate(names) if n.endswith(".effort")]
    vel_i = [i for i, n in enumerate(names) if n.endswith(".vel")]
    pos_i = [i for i, n in enumerate(names) if n.endswith(".pos")]

    print(f"\n=== 스키마 ===")
    print(f"  observation.state 차원 : {len(names)}")
    print(f"  pos {len(pos_i)}개 / effort {len(eff_i)}개 / vel {len(vel_i)}개")
    print(f"  action 차원            : {info['features']['action']['shape'][0]}")
    print(f"  fps={info['fps']}  에피소드={info.get('total_episodes','?')}  프레임={info.get('total_frames','?')}")

    if not eff_i:
        print(f"\n{BAD} effort가 없습니다 — USE_EFFORT가 반영 안 된 상태로 녹화됨{END}")
        print("   → recording.env의 USE_EFFORT=true 확인")
        print("   → GUI에서 Command 창에 --robot.use_effort=true 가 있는지 확인")
        print("   → 셸 스크립트(5__record.sh)로 찍었다면 effort는 절대 안 들어감")
        return 1
    print(f"\n{OK} effort 필드 존재{END}")

    if len(names) != 20:
        print(f"{WARN} 차원이 20이 아님({len(names)}) — 예상 구성과 다름{END}")

    # 실제 값 확인 (차원만 맞고 값이 전부 0인 경우를 잡는다)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        print(f"{BAD} data parquet 없음{END}")
        return 1
    t = pq.read_table(files[0])
    state = np.stack(t.column("observation.state").to_pylist())

    eff = state[:, eff_i]
    vel = state[:, vel_i]

    print(f"\n=== 값 확인 ({files[0].name}, {len(state)} 프레임) ===")
    print(f"  effort |max|  : {np.abs(eff).max():.3f} N·m")
    print(f"  effort 표준편차: {eff.std():.4f}")
    print(f"  vel    |max|  : {np.abs(vel).max():.3f}")

    rc = 0
    if np.abs(eff).max() == 0:
        print(f"\n{BAD} effort가 전부 0 — CAN에서 값을 못 읽고 있음{END}")
        print("   → 팔 전원/CAN 연결 확인. 스키마만 맞고 데이터는 쓸모없는 상태")
        rc = 1
    elif eff.std() < 1e-6:
        print(f"\n{BAD} effort가 상수 — 센서 읽기가 갱신되지 않음{END}")
        rc = 1
    else:
        print(f"\n{OK} effort 값이 실제로 변하고 있음{END}")

    # 관절별 최댓값 → SAFETY_EFFORT_LIMIT 튜닝 근거
    print(f"\n=== 관절별 |effort| 최댓값 (SAFETY_EFFORT_LIMIT 튜닝용) ===")
    for i, n in zip(eff_i, [names[i] for i in eff_i]):
        print(f"  {n:18s} {np.abs(state[:, i]).max():7.3f} N·m")
    peak = np.abs(eff).max()
    print(f"\n  → 이 동작이 '정상 지우기'였다면 SAFETY_EFFORT_LIMIT ≈ {peak*1.5:.1f} 권장")

    # smooth start 오염 검사
    print(f"\n=== Smooth Start 오염 검사 ===")
    ep = t.column("episode_index").to_numpy(zero_copy_only=False)
    fi = t.column("frame_index").to_numpy(zero_copy_only=False)
    first_ep = state[(ep == ep[0])][np.argsort(fi[ep == ep[0]])]
    head = np.abs(first_ep[:100, eff_i])
    tail = np.abs(first_ep[100:, eff_i]) if len(first_ep) > 100 else head
    if len(first_ep) > 120 and head.std() > 0 and tail.std() > 0:
        # 보정되면 초반이 부자연스럽게 매끈한 선형 구간이 된다
        lin = np.abs(np.diff(first_ep[:100, eff_i], n=2, axis=0)).max()
        if lin < 1e-4:
            print(f"{BAD} 초반 100프레임 effort가 선형 — Smooth Start에 덮어써짐{END}")
            print("   → SMOOTH_START_FRAMES=0 으로 끄고 재녹화 필요")
            rc = 1
        else:
            print(f"{OK} 초반 프레임 effort 정상 (덮어쓰기 흔적 없음){END}")
    else:
        print(f"{WARN} 프레임이 적어 판정 생략 (120프레임 이상 필요){END}")

    return rc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
