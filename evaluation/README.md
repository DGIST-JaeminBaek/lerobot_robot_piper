# evaluation/

평가 세션 결과가 쌓이는 곳. 세션마다 `<MMDD>_<model>_<condition>/` 폴더가 생긴다.

```
evaluation/
  0818_smolvla_async/
    episodes.csv    에피소드 1행씩 (원자료 — 이걸 지우면 다시 못 만든다)
    summary.md      조건별 요약표. 보고서에 그대로 붙일 수 있다
    summary.json    같은 내용, 스크립트/AI 입력용
  manual_sheet_template.csv   도구가 안 될 때 쓰는 손 채점 시트
```

사용법과 채점 기준은 [docs/policy/evaluation_protocol.md](../docs/policy/evaluation_protocol.md).

빠른 시작:

```bash
DRY_RUN='<데이터셋>/erase_the_rectangle_*' bash scripts/13__eval_session.sh   # 로봇 없이 창 점검
python scripts/tools/erase_eval.py --summarize evaluation/<폴더>/episodes.csv  # 요약만 다시 만들기
```
