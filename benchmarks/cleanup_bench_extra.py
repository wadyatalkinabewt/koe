"""Tiebreaker (N=3 for top 3) + no-glossary sanity check for winner."""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cleanup_bench import MODELS, SAMPLES, GLOSSARY, call_model

API_KEY = os.environ["OPENROUTER_API_KEY"]


def main():
    bench_dir = Path(__file__).parent
    top3 = ["gemini-3.5-flash", "sonnet-4-6", "deepseek-v3.2"]
    top3_models = [m for m in MODELS if m["key"] in top3]

    # Tiebreaker: 2 additional runs (run_idx 1 and 2) for top 3 = 90 calls
    tasks = []
    for run_idx in (1, 2):
        for m in top3_models:
            for s in SAMPLES:
                tasks.append((m, s, GLOSSARY, run_idx))

    print(f"Tiebreaker: {len(tasks)} calls (3 models x 15 samples x 2 extra runs)")
    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(call_model, m, s, gl, ri) for m, s, gl, ri in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)

    results.sort(key=lambda r: (r["model"], r["sample_id"], r["run_idx"]))
    out_tb = bench_dir / "results_tiebreaker.jsonl"
    with out_tb.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} tiebreaker results to {out_tb}")

    # Sanity: current cleanup candidate on no-glossary regime, 15 calls
    winner_model = next(m for m in MODELS if m["key"] == "gemini-3.5-flash")
    print(f"\nNo-glossary sanity check: {len(SAMPLES)} calls for {winner_model['key']}")
    sanity = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(call_model, winner_model, s, "(no glossary provided)", 99) for s in SAMPLES]
        for fut in as_completed(futures):
            sanity.append(fut.result())

    sanity.sort(key=lambda r: r["sample_id"])
    out_san = bench_dir / "results_no_glossary.jsonl"
    with out_san.open("w", encoding="utf-8") as f:
        for r in sanity:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sanity)} no-glossary results to {out_san}")


if __name__ == "__main__":
    main()
