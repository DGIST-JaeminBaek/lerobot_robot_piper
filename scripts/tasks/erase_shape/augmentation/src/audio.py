from __future__ import annotations

"""증강 영상에 원본 오디오를 다시 붙이는 선택 기능.

현재 생성한 0802~0805 증강 결과는 오디오가 핵심이 아니므로 기본적으로 사용하지 않았다.
"""

import shutil
import subprocess
from pathlib import Path


def mux_original_audio(silent_video: Path, input_video: Path, output_video: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True
