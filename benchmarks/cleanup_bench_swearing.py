"""Swearing spot-check — gemini-3.5-flash, locked production prompt."""

import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cleanup_bench import GLOSSARY, PROMPT, MODELS

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"

WINNER = next(m for m in MODELS if m["key"] == "gemini-3.5-flash")

SWEARING_SAMPLES = [
    {"id": "A", "domain": "tradie-frustrated",
     "raw": "fuck me the fucken excavator's shit itself again, bloody thing's only six months old. ring grant from cat and tell him I want a loaner here by tomorrow morning or we're rooted for the Hewick pour. and if he tries to bullshit me about availability tell him to get fucked we've spent eighty grand with them this year"},
    {"id": "B", "domain": "tradie-cword",
     "raw": "hey pax the boys just cracked into a beer at smoko, tell tony that mad cunt to bring his ute round for the timber pickup. he's a good cunt, just sometimes forgets shit. also let dave know the friggen invoice is still missing, he keeps fobbing me off the prick"},
    {"id": "C", "domain": "tradie-meta-instruction",
     "raw": "fucks sake the council just kicked back the resource consent for the fucken third time, this time they're whinging about stormwater. absolute clown shoes. anyway pull up the original consent application from september, find the stormwater section, and email it to the consultant engineer at hayes design with a polite note saying we've already submitted this twice. politely though, dont put any of this venting in the email"},
]


def call(sample, run_idx=0):
    body = {
        "model": WINNER["id"],
        "messages": [{"role": "user", "content": PROMPT.format(glossary=GLOSSARY, transcription=sample["raw"])}],
        "temperature": 0.0,
        "max_tokens": 2000,
        "provider": {"order": WINNER["providers"], "allow_fallbacks": False},
    }
    t0 = time.perf_counter()
    r = requests.post(URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=body, timeout=120)
    elapsed = (time.perf_counter() - t0) * 1000
    j = r.json()
    choice = (j.get("choices") or [{}])[0]
    return {
        "sample_id": sample["id"],
        "domain": sample["domain"],
        "run_idx": run_idx,
        "elapsed_ms": round(elapsed, 1),
        "status_code": r.status_code,
        "output": choice.get("message", {}).get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": j.get("usage", {}),
        "provider": j.get("provider"),
        "error": j.get("error"),
        "raw_response": j if j.get("error") or not choice.get("message", {}).get("content") else None,
    }


if __name__ == "__main__":
    # N=3 each so refusal/sanitisation can be detected as stochastic vs deterministic
    tasks = [(s, run_idx) for s in SWEARING_SAMPLES for run_idx in range(3)]
    print(f"Running {len(tasks)} calls (3 samples x 3 runs)...")

    results = []
    with ThreadPoolExecutor(max_workers=9) as ex:
        futures = [ex.submit(call, s, ri) for s, ri in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (r["sample_id"], r["run_idx"]))
    out_path = Path(__file__).parent / "results_swearing.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} results to {out_path}\n")

    # Quick analysis: did any swears get stripped?
    swear_words = ["fuck", "shit", "bloody", "rooted", "bullshit", "cunt", "friggen", "prick", "whinging", "clown shoes"]
    for sid in ["A", "B", "C"]:
        runs = [r for r in results if r["sample_id"] == sid]
        print(f"=== Sample {sid} ({runs[0]['domain']}) ===")
        for r in runs:
            output_lower = r["output"].lower()
            swears_found = [w for w in swear_words if w in output_lower]
            refused = "refus" in output_lower or "cannot" in output_lower or "i'm sorry" in output_lower or not r["output"].strip()
            tag = "REFUSED" if refused else f"swears_kept={len(swears_found)}"
            print(f"  run{r['run_idx']} [{r['elapsed_ms']:>5.0f}ms] [{tag}]")
            print(f"    {r['output'][:300]}")
        print()
