"""
Cleanup model benchmark — Koe + Acme pipeline.

Tests 6 candidate cleanup models against 15 voice-transcription samples
spanning Alex's snippets and 10 Acme client domains (tradie, te reo,
broker, NZ accents, brand recognition, structured data, self-correction,
acronym recovery, reported speech, NZ-isms).

Single regime: adversarial-irrelevant glossary (~50 entries, most don't
appear in any single sample). N=1 main pass.
"""

import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    {"key": "haiku-4-5",       "id": "anthropic/claude-haiku-4-5",            "providers": ["Anthropic"]},
    {"key": "sonnet-4-6",      "id": "anthropic/claude-sonnet-4-6",           "providers": ["Anthropic"]},
    {"key": "gemini-3.1-lite", "id": "google/gemini-3.1-flash-lite-preview",  "providers": ["Google AI Studio"]},
    {"key": "gemini-3-flash",  "id": "google/gemini-3-flash-preview",         "providers": ["Google AI Studio"]},
    {"key": "gpt-5.4-mini",    "id": "openai/gpt-5.4-mini",                   "providers": ["OpenAI"]},
    {"key": "deepseek-v3.2",   "id": "deepseek/deepseek-v3.2",                "providers": ["Friendli"]},
]

GLOSSARY = """\
Construction & NZ trades:
- Howick: suburb in east Auckland (sometimes misheard as "Hewick")
- BCA: Building Consent Authority
- Bunnings: hardware retailer (Bunnings Warehouse, NZ/AU)
- Carters: NZ building supplies merchant
- GIB: NZ plasterboard brand (Winstone Wallboards) — sometimes misheard as "give board"
- Glenfield: suburb in north Auckland
- Naylor Love: major NZ construction company — sometimes misheard as "nailor love"
- Fletcher Construction: largest NZ construction company
- COC: Certificate of Compliance
- BIM coordinator: Building Information Modeling coordinator — sometimes misheard as "beam coordinator"
- RFI: Request for Information
- PCBU: Person Conducting a Business or Undertaking (NZ H&S term) — sometimes misheard as "pay-see-boo"
- Site Safe: NZ construction safety organisation
- EOT: Extension of Time
- NCR: Non-Conformance Report
- Aldridge: NZ scaffolding company
- scaff: scaffolding (informal)
- tagged: scaffold tag (inspection certificate)
- H3 90 by 45: structural framing timber spec (90mm x 45mm, H3 treatment)

Insurance & NZ business:
- TrimFix: example client name (proper noun) — sometimes misheard as "trim fix"
- Vero: NZ insurance company
- IAG: Insurance Australia Group
- PI: Professional Indemnity insurance
- CoFI: Conduct of Financial Institutions Act (NZ regulatory) — sometimes misheard as "co-fi"
- FAP: Financial Advice Provider (NZ regulatory)
- D&O: Directors and Officers insurance (write as "D&O" not "D and O")
- Giltrap: NZ vehicle dealership chain — sometimes misheard as "guilt trap"
- G.J. Gardner: NZ home builder — sometimes misheard as "GG Gardner"
- Mike Greer: Mike Greer Homes (NZ builder, capitalize)
- Jennian: Jennian Homes (NZ builder) — sometimes misheard as "Jennings"
- NZBN: New Zealand Business Number (13 digits)
- Punahau: street name (Papakura)
- Papakura: south Auckland suburb

Te reo Maori:
- kia ora: greeting
- hui: meeting/gathering
- powhiri: formal welcome ceremony — sometimes misheard as "powery"
- Pukekohe: town south of Auckland — sometimes misheard as "puke-koh-hee"

NZ idioms (preserve untouched):
- yeah nah: idiom for "no"
- sweet as: idiom for "good/fine"
- she'll be right: idiom for "it'll be fine"
- ute: utility vehicle/pickup truck
- "tell em": "tell them" (preserve as is)
- "job's a good un": idiom for "good work"

Tech / AI ops (Acme context):
- Acme: SMB AI agent business — sometimes misheard as "Ack Me", "kreaka", "crikey"
- Hetzner: VPS hosting provider — sometimes misheard as "Hesner", "Hetsner"
- Hermes: agent runtime — sometimes misheard as "Hermez", "Hermes" (NOT "Hermes" the brand)
- Civis: knowledge base for AI agents (app.civis.run) — sometimes misheard as "Civic", "CVS"
- Koe: transcription tool (pronounced ko-eh)
- Supabase: backend-as-a-service — sometimes misheard as "sopa-base"
- pgvector: Postgres vector extension
- OAuth: authentication standard — sometimes misheard as "o-arth"
- MCP: Model Context Protocol
- OpenRouter: AI model routing service — sometimes misheard as "opener router"
- Qwen: Alibaba LLM — sometimes misheard as "Quinn"
- GLM: Zhipu LLM (sometimes spoken as "G-L-M")
- Gemma: Google open-source LLM
- Minimax: Chinese AI lab — sometimes misheard as "Mini Max"
- Claude Code: Anthropic CLI agent
- Codex: OpenAI coding agent
- systemd: Linux service manager
- cron: Linux scheduler
- VPS: Virtual Private Server
- JSONL: JSON Lines format (write as "JSONL" not "JSON L")
- SSE: Server-Sent Events
- CLAUDE.md, AGENTS.md, soul.md: agent instruction files
"""

PROMPT = """\
Lightly clean up this voice transcription. Preserve all content — including conversational wrap-ups, asides, and trailing sentences. The user's voice and intent must come through.

Rules:
- Add punctuation, capitalization, and paragraph breaks
- Remove standalone disfluencies: "um", "uh", "ah", "hmm", and stuttered repeats like "I-I-I think" -> "I think"
- Fix clearly misheard technical terms using the glossary below
- Keep semantic fillers ("like", "you know", "I mean", "kind of", "I think") — they carry tone
- Keep regional/colloquial language untouched (NZ-isms, te reo Maori, idioms, swearing)
- Do NOT summarize, paraphrase, or add information that wasn't said
- Do NOT drop any content, including trailing wrap-ups
- For self-corrections, follow the LATER instruction (the speaker overrode the earlier one)
- Normalize spoken numbers/addresses/emails into standard written form (e.g. "oh nine" -> "09", "dot co dot enzed" -> ".co.nz")

Glossary (terms that may appear; only correct if the audio clearly intended one of these — most entries will not appear in this transcription):

{glossary}

Output ONLY the cleaned plain text. No markdown, no bullet points, no annotations like "(fixed: X->Y)", no preamble, no quotes wrapping the output.

Transcription:
{transcription}
"""

SAMPLES = [
    {"id": 1, "domain": "Alex-wrap-up",
     "raw": "yeah so um the Ack Me chat service is hitting rate limits on Hesner so I think we need to scale that up to a CPX42 or maybe just add a second VPS honestly. Hermez can handle the load no problem its just the network. okay cool get on with it"},
    {"id": 2, "domain": "Alex-domain",
     "raw": "I want to add a new MCP tool to civic that pulls build logs filtered by stack tag so when claude code is searching it can do explore endpoint with say like nextjs and supabase tags and pgvector. then we wire that into opener router via the mcp server at mcp.civic.run"},
    {"id": 3, "domain": "Alex-fillers",
     "raw": "um so like, I think the thing is, you know, we kind of need to, like, move the cron jobs off the main VPS because, I mean, theyre eating CPU when hermes is doing its thing. so like, maybe a separate worker, you know, like a cloudflare worker or something, I dont know. yeah"},
    {"id": 4, "domain": "Alex-models",
     "raw": "so I-I-I tested Quinn 32B against G-L-M 4.5 and Mini Max for the agent runtime and honestly Quinn was best for tool calling but G-L-M won on cost. Codex from openai uh did okay but its expensive. anyway lets just go with Quinn for now"},
    {"id": 5, "domain": "Alex-long",
     "raw": "ok so for the Acme admin dashboard we need an o-arth flow because right now operators are just using a shared password which is uh really bad. ideally we use sopa-base auth with magic links. then for the fleet API we need to add a JSON L endpoint that streams build logs as they come in so the chat service can tail them in real time over SSE. yeah I think that covers it let me know if you have questions"},
    {"id": 6, "domain": "tradie",
     "raw": "yeah na so we're framed up at the Hewick job and the bca wants the producer statement before we can pour the slab next week. Bunning's is out of H3 90 by 45 again so I called carters and they said Tuesday at the earliest. tell the boys to grab the give board off the Glen Field site cause we've got extra"},
    {"id": 7, "domain": "te-reo",
     "raw": "kia ora team just a quick one I'm jumping on a hui with the fletchers crew at three then a powery at puke-koh-hee should be back at the office around five. anyway can someone get the C-O-C sorted for the nailor love project and email it to Sarah at I-A-G before close of business"},
    {"id": 8, "domain": "broker-jargon",
     "raw": "okay so the renewal for trim fix is up next month and we need to get the schedule across to vero by Friday. their P-I went up ten percent last year because of the FAP regs and Co-fi requirements. I reckon we should also offer them a chat with Casey about adding D and O cover since they just brought on two new directors. let me know what you think"},
    {"id": 9, "domain": "vowel-shift",
     "raw": "we need to ship the tinder before the fifteenth, customer's been waiting six weeks already. quote was eighteen K but they're trying to push it to eighty. tell em drop the price by ten percent and we close it today, if not we walk"},
    {"id": 10, "domain": "nz-brands",
     "raw": "Casey reckons the guilt trap deal is on, they want forty utes for the Christchurch crew before end of June. send the proposal tonight. while you're at it, the GG Gardner thing is sorted, mike greer is back in, and Jennings doesn't want to play ball this round"},
    {"id": 11, "domain": "structured-data",
     "raw": "right send the invoice to oh nine triple three two oh six four for stevens building they're at thirty two punahau road papakura, NZBN nine four two nine oh four six eight oh oh one six oh. GST inclusive total is twelve thousand six hundred and eighty four ninety. due on the twentieth"},
    {"id": 12, "domain": "self-correction",
     "raw": "okay so send the contract to Casey at trim fix actually no wait send it to Sarah she's the one signing now Casey's just CCing. um yeah Sarah at sarah dot taylor at trim fix dot co dot enzed. and tell her we need it back by Thursday end of day not Friday like I said earlier"},
    {"id": 13, "domain": "acronym-recovery",
     "raw": "the pay-see-boo wants their site safe audit done before the EOT request goes in. our beam coordinator can't get to the model until Tuesday so flag the rfi to the architect and tell them we need turnaround in three days. NCR closeout is overdue too"},
    {"id": 14, "domain": "reported-speech",
     "raw": "so Tony from Aldridge rang and said and I quote his exact words our boys won't be on site Monday because the scaff isn't tagged yet end quote. so we need to push back the inspection. tell the inspector Tony said it's a one day delay max"},
    {"id": 15, "domain": "nz-isms",
     "raw": "yeah nah it's all sweet as bro, the consent came through this morning. she'll be right for the pour Friday. tell the crew sweet as keep going. job's a good un"},
]


def call_model(model, sample, glossary=GLOSSARY, run_idx=0):
    """Single cleanup call. Returns dict with output, latency, usage."""
    prompt_text = PROMPT.format(glossary=glossary, transcription=sample["raw"])
    body = {
        "model": model["id"],
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": 2000,
        "provider": {"order": model["providers"], "allow_fallbacks": False},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            j = r.json()
        except Exception:
            j = {"_raw": r.text[:500]}
        choice = (j.get("choices", [{}]) or [{}])[0] if isinstance(j, dict) else {}
        return {
            "model": model["key"],
            "model_id": model["id"],
            "sample_id": sample["id"],
            "domain": sample["domain"],
            "run_idx": run_idx,
            "elapsed_ms": round(elapsed_ms, 1),
            "status_code": r.status_code,
            "output": choice.get("message", {}).get("content", "") if isinstance(j, dict) else "",
            "finish_reason": choice.get("finish_reason") if isinstance(j, dict) else None,
            "usage": j.get("usage", {}) if isinstance(j, dict) else {},
            "provider": j.get("provider") if isinstance(j, dict) else None,
            "error": j.get("error") if isinstance(j, dict) else None,
        }
    except Exception as e:
        return {
            "model": model["key"],
            "model_id": model["id"],
            "sample_id": sample["id"],
            "domain": sample["domain"],
            "run_idx": run_idx,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
            "error": str(e),
        }


def run_main_pass(out_path):
    tasks = [(m, s) for m in MODELS for s in SAMPLES]
    print(f"Running {len(tasks)} cleanup calls in parallel (max 20 workers)...")

    results = []
    done_count = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(call_model, m, s): (m["key"], s["id"]) for m, s in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            done_count += 1
            results.append(r)
            err = r.get("error")
            tag = "OK"
            if err:
                tag = f"ERR ({str(err)[:50]})"
            print(f"  [{done_count:>3}/{len(tasks)}] {r['model']:<20} sample {r['sample_id']:>2} ({r.get('elapsed_ms',0):>6.0f}ms) {tag}")

    results.sort(key=lambda r: (r["model"], r["sample_id"]))

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} results to {out_path}")
    return results


def summary(results):
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    print("\nLatency + cost summary (per model):")
    print(f"  {'model':<20} {'P50_ms':>8} {'P95_ms':>8} {'avg_in':>8} {'avg_out':>8} {'avg_$':>10} {'errors':>7}")
    for model_key, rs in by_model.items():
        ok = [r for r in rs if not r.get("error") and r.get("usage", {}).get("total_tokens")]
        if not ok:
            print(f"  {model_key:<20} ALL ERRORED")
            continue
        latencies = sorted([r["elapsed_ms"] for r in ok])
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        avg_in = sum(r["usage"].get("prompt_tokens", 0) for r in ok) / len(ok)
        avg_out = sum(r["usage"].get("completion_tokens", 0) for r in ok) / len(ok)
        avg_cost = sum(r["usage"].get("cost", 0) for r in ok) / len(ok)
        errs = sum(1 for r in rs if r.get("error"))
        print(f"  {model_key:<20} {p50:>8.0f} {p95:>8.0f} {avg_in:>8.0f} {avg_out:>8.0f} {avg_cost:>10.5f} {errs:>5}/{len(rs)}")


if __name__ == "__main__":
    out_path = Path(__file__).parent / "results_main.jsonl"
    results = run_main_pass(out_path)
    summary(results)
