#!/usr/bin/env bash
# 바탕화면에 실행 스크립트(피퍼실행.sh), 앱 메뉴에 "Piper Teleop GUI" 항목을 등록.
# 두 템플릿의 %REPO_DIR% 을 이 레포의 실제 경로로 치환해서 설치함.
#
# 바탕화면은 .desktop 파일 대신 순수 쉘 스크립트를 씀 — Nautilus가 .desktop 파일을
# "신뢰"로 인식하는 게 안정적이지 않아 더블클릭해도 텍스트 에디터로 열리는 경우가
# 있었음. 순수 .sh는 더블클릭 시 Nautilus가 Run/Run in Terminal/Display 선택
# 대화상자를 띄우므로 "Run in Terminal"을 고르면 확실하게 동작함.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DESKTOP_SH_TEMPLATE="${SCRIPT_DIR}/piper-launch.sh.template"
DESKTOP_SH_NAME="피퍼실행.sh"
DESKTOP_ENTRY_TEMPLATE="${SCRIPT_DIR}/piper-teleop-gui.desktop.template"
DESKTOP_ENTRY_NAME="piper-teleop-gui.desktop"

# 1) ~/Desktop 에 실행 스크립트 설치
mkdir -p "${HOME}/Desktop"
desktop_sh_path="${HOME}/Desktop/${DESKTOP_SH_NAME}"
sed "s#%REPO_DIR%#${REPO_DIR}#g" "${DESKTOP_SH_TEMPLATE}" > "${desktop_sh_path}"
chmod +x "${desktop_sh_path}"
if command -v gio >/dev/null 2>&1; then
  gio set "${desktop_sh_path}" metadata::trusted true 2>/dev/null || true
fi
echo "[OK] ${desktop_sh_path}"

# 2) 앱 메뉴(~/.local/share/applications)에는 .desktop 파일로 등록
#    (Nautilus 더블클릭이 아니라 GNOME Activities 검색으로 실행하므로 trust 이슈 없음)
mkdir -p "${HOME}/.local/share/applications"
entry_path="${HOME}/.local/share/applications/${DESKTOP_ENTRY_NAME}"
sed "s#%REPO_DIR%#${REPO_DIR}#g" "${DESKTOP_ENTRY_TEMPLATE}" > "${entry_path}"
chmod +x "${entry_path}"
if command -v gio >/dev/null 2>&1; then
  gio set "${entry_path}" metadata::trusted true 2>/dev/null || true
fi
echo "[OK] ${entry_path}"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true
fi

echo
echo "완료."
echo "- 바탕화면: '${DESKTOP_SH_NAME}' 더블클릭 → 'Run in Terminal' 선택"
echo "- 앱 메뉴(Activities 검색): 'Piper Teleop GUI' 검색해서 실행"
