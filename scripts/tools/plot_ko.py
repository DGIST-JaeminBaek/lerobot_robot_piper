"""matplotlib 한글 폰트 설정 — 그림 만드는 도구들이 공용으로 쓴다.

랩 PC에 NanumGothic이 없고 Noto Sans CJK만 깔려 있다. matplotlib의 폰트 목록에는
'Noto Sans CJK JP'로만 잡히는데(KR/JP/SC/TC가 한 파일에서 나온 서브셋이라 이름이
하나로 등록된다), 이 폰트는 한글 글리프를 전부 포함하므로 그대로 쓰면 된다.
설정을 안 하면 제목·축 라벨의 한글이 전부 두부(□)로 나온다.
"""

import matplotlib

matplotlib.use("Agg")  # 헤드리스 — import 순서상 pyplot보다 먼저여야 한다
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 앞에 있는 것부터 시도한다. 다른 PC에서 NanumGothic이 있으면 그쪽이 먼저 잡힌다.
CANDIDATES = ("NanumGothic", "Malgun Gothic", "AppleGothic",
              "Noto Sans CJK KR", "Noto Sans CJK JP")


def use_korean() -> str | None:
    """설치된 한글 폰트를 rcParams에 물린다. -> 고른 폰트 이름 (없으면 None)"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            # 한글 폰트로 바꾸면 마이너스 기호가 깨진다 — 유니코드 대신 ASCII를 쓴다
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


__all__ = ["use_korean", "plt"]
