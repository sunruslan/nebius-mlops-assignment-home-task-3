# Report: Qwen3-30B-A3B Text-to-SQL PoC

**Hardware:** 1× NVIDIA H100 80GB  
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`  
**Target SLO:** P95 end-to-end agent latency &lt; 5s, 10+ RPS over a 5-minute window

---

## 1. Serving configuration (Phase 1)

Config lives in `infra/vllm_config.yaml`, launched via `scripts/start_vllm.sh`.

| Flag | Value | Justification |
|---|---|---|
| `model` / `served-model-name` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Fixed assignment model; served name matches `VLLM_MODEL` in `.env`. |
| `max-model-len` | `4096` | Model defaults to 262K context, which OOM'd on startup (~24 GiB KV needed vs ~12.7 GiB available). Agent prompts are 1.5–3K tokens + short SQL/JSON outputs; capping context frees KV slots for concurrency. Tuned down from 8192 → 6144 → 4096 during load testing when KV cache hit 99%. |
| `gpu-memory-utilization` | `0.98` | MoE weights consume ~57 GiB; pushing utilization maximizes KV cache for concurrent sequences. |
| `enable-chunked-prefill` | `true` | Schema-heavy agent prefills are 1–3K tokens; chunking avoids long prefill steps blocking decode under load. |
| `max-num-batched-tokens` | `8192` | Allows batching multiple agent prefill requests per scheduler step. |
| `max-num-seqs` | `512` | Headroom for 10 agent RPS × ~2–3 vLLM calls/request, plus revise-loop bursts. |
| `enable-prefix-caching` | `true` | Agent re-sends identical system prompt + DB context across `generate_sql` / `verify` / `revise`; prefix cache avoids recomputing shared tokens. |
| `dtype` | `auto` | H100-native bf16 for the MoE model without manual dtype tuning. |
| `trust-remote-code` | `true` | Required for Qwen model loading in vLLM. |

**Startup fix:** Initial launch failed with `ValueError` on KV cache (262144 default context). Setting `max-model-len: 8192` allowed the model to load; further tuning reduced it to 4096 under sustained load.

**Agent serving:** The agent runs separately on port 8001 (`scripts/start_agent.sh`, default 48 uvicorn workers). A single-worker agent was the first load-test bottleneck (see §3).

---

## 2. Observability (Phase 2)

Grafana dashboard: `infra/grafana/provisioning/dashboards/serving.json`.

Three panel groups, all fed from vLLM `/metrics` via Prometheus:

- **Latency** — E2E / TTFT / per-phase (queue, prefill, decode) / TPOT percentiles. Answers *where* in the request lifecycle time is spent.
- **Throughput** — Running & waiting requests, completed req/s, generation & prefill tokens/s, tokens-per-request P95.
- **KV cache** — Usage %, prefix-cache hit rate, preemptions/s, cached vs computed prefill tokens.

During the Phase 5 eval run (~30 agent requests over 83s), panels reacted as expected (requests running, token rates, latency histograms). Screenshot: `screenshots/grafana_eval_run.png`.

---

## 3. SLO tuning attempts (Phase 6 — partial)

Phase 6 was not completed to a passing SLO. Below is an honest log of what was tried, grounded in Grafana/Prometheus observations.

**SLO target:** P95 agent latency &lt; 5s at 10 RPS for 300s.

### Baseline load test

`load_test/driver.py --rps 10 --duration 300` with default single-worker agent + initial vLLM config (`max-model-len: 8192`).

| Metric | Result |
|---|---|
| OK / 3000 requests | 128 (4.3%) |
| P95 latency | **112.6s** |
| Timeouts | 2206 |

Prometheus during this run: vLLM `num_requests_waiting = 0`, KV cache &lt; 35%, vLLM E2E P95 ≈ 4s. **vLLM was not the bottleneck** — the single-threaded agent server queued thousands of requests.

> **saw** agent P95 112s with 96% request failures and vLLM queue P95 0.29s → **hypothesized** agent serialization, not vLLM saturation → **changed** agent to 32 uvicorn workers → **result** 52% OK on 60s smoke test, P95 still 63.5s.

### Iteration 1 — agent concurrency

32 uvicorn workers (`--workers 32`). 60s smoke at 10 RPS: 310/600 OK, P95 63.5s. Improvement, but still far from SLO.

### Iteration 2 — vLLM + agent tuning

vLLM: `max-model-len: 6144`, `max-num-batched-tokens: 16384`, `max-num-seqs: 512`. Agent: 48 workers.

300s run at 10 RPS: **1727/3000 OK**, P95 **69.6s**, 561 timeouts.

Prometheus: **KV cache peaked at 99.3%**, vLLM E2E P95 rose to **9.8s**. Queue time still low (~0.29s) — the system was memory-bound on KV slots, not scheduler-queue-bound.

> **saw** KV cache at 99% and vLLM E2E P95 9.8s under parallel agent load → **hypothesized** too many concurrent sequences for available KV blocks at `max-model-len: 6144` → **changed** `max-model-len: 4096`, `gpu-memory-utilization: 0.98` → **result** final 300s run interrupted; 5 RPS probe still showed P95 30.7s.

### Verdict

| Criterion | Target | Best observed |
|---|---|---|
| P95 agent latency | &lt; 5s | 11.4s (iter 2, successful requests only) |
| Sustained RPS | 10+ | ~4.8 effective throughput at 10 RPS offered load |
| Request success rate | ~100% | 58% (iter 2) |

**Root causes identified:**
1. **Agent concurrency** — single-worker uvicorn cannot serve 10 RPS when each agent run takes 1–3+ seconds (2–3 sequential LLM calls).
2. **KV cache saturation** — once agent concurrency was fixed, KV cache usage hit 99%, degrading per-request vLLM latency even with low scheduler queue depth.

`results/eval_after_tuning.json` was not produced; eval numbers below are from the baseline configuration run in Phase 5.

---

## 4. Baseline eval results (Phase 5)

Run: `python evals/run_eval.py --out results/eval_baseline.json`  
30 questions from `evals/eval_set.jsonl`, execution-accuracy scoring (canonicalized row sets).

| Metric | Value |
|---|---|
| Overall pass rate | **46.7%** (14/30) |
| Iteration 0 pass rate | 43.3% (13/30) |
| Iteration 1 pass rate | 46.7% (14/30) |
| Iterations 2–3 | 46.7% (14/30) |
| Avg agent iterations | 1.17 |
| Gold SQL failures | 0 |
| Agent request failures | 0 |

---

## 5. Agent value

The verify→revise loop provides a **modest but real** quality gain. Per-iteration pass rate rises from 43.3% at iteration 0 to 46.7% at iteration 1 (+1 question: e.g. `card_games` "Ancestor's Chosen" and `thrombosis_prediction` IgG-level question were corrected after revise). No further gains at iterations 2–3 — either the first revision fixes the issue or the loop exhausts its budget without recovery.

The loop is not free: revised questions average more LLM calls (up to 4 iterations), which matters for latency under load. For this workload, the architecture earns its keep on hard schema/semantic mismatches but is not a large quality multiplier on the full 30-question set.

---

## 6. What I'd do with more time

1. **Right-size agent concurrency** — run a short sweep of uvicorn workers (16/32/64/96) vs achieved RPS at fixed vLLM config; pick the knee of the latency curve rather than guessing.
2. **KV-cache-aware vLLM sizing** — binary-search `max-model-len` (4096–8192) while watching `vllm:kv_cache_usage_perc` and `vllm:e2e_request_latency_seconds` P95 under 10 RPS until KV stays below ~85%.
3. **Re-run eval after tuning** — produce `results/eval_after_tuning.json` to confirm quality didn't regress when serving config changes (especially lower `max-model-len`, which could truncate long schema contexts).
4. **Agent-side latency** — explore parallelizing the three exploration nodes (`explore_counts` / `explore_samples` / `explore_categoricals`) since they are independent SQLite reads and add ~200ms before the first LLM call.
5. **Prompt/token budget** — trim categorical profiles for large BIRD databases to stay safely under `max-model-len` without sacrificing the columns the model needs for filters.

---

## Artifacts

| File | Description |
|---|---|
| `infra/vllm_config.yaml` | Final vLLM serving config |
| `infra/grafana/provisioning/dashboards/serving.json` | Grafana dashboard |
| `agent/graph.py`, `agent/prompts.py` | Text-to-SQL agent with verify→revise loop |
| `evals/run_eval.py` | Execution-accuracy eval runner |
| `results/eval_baseline.json` | Baseline eval (46.7% pass rate) |
| `results/load_test_*.json` | Partial Phase 6 load-test runs |
| `scripts/start_agent.sh` | Agent launcher with configurable workers |
| `screenshots/grafana_eval_run.png` | Grafana during baseline eval |
