# ShopAssist cost optimization: solution

## Results at a glance

| | Baseline (avg / public ticket) | Ours (avg / public ticket) | Target | Saved |
|---|---:|---:|---:|---:|
| **Q1** single call | 8,864 | **1,910** | ≤ 3,000 | 78.5% |
| **Q2** pipeline | 73,538 | **1,607** | ≤ 2,500 | 97.8% (≈ 46× cheaper) |

All four public tickets (T-1001 to T-1004) and the holdout T-1091 keep the correct `action` and `kb_article` in both tracks.

### How these numbers were measured
- Every figure comes from `CostSession` (`shared/billing.py`), so it uses the repository's own token counting (tiktoken `cl100k_base`) and pricing (input 3, cached input 0.3, output 15).
- Baseline numbers come from re-running verbatim copies of the original `question1/agent.py` and `question2/agent.py` side by side with the new code, on the same tickets in the same run.
- The baseline's verbose completion contains a random UUID, so its points vary by roughly ±100 from run to run. The optimized versions produce the same numbers on every run.
---

## Question 1: single call

### Per-ticket results

| Ticket | Baseline points | Ours: input tok | Ours: output tok | Ours: points |
|---|---:|---:|---:|---:|
| T-1001 | 8,814 | 447 | 46 | **2,031** |
| T-1002 | 8,976 | 441 | 39 | **1,908** |
| T-1003 | 8,724 | 428 | 39 | **1,869** |
| T-1004 | 8,943 | 430 | 36 | **1,830** |
| T-1091 (holdout) | 8,946 | 434 | 38 | **1,872** |
| **Public total** | **35,457** | | | **7,638** |

### Per-call breakdown: T-1001 (one LLM call)

| | Input tok | Output tok | Input pts | Output pts | Total |
|---|---:|---:|---:|---:|---:|
| Baseline `handle_ticket` | 523 | 483 | 1,569 | 7,245 | 8,814 |
| Ours `handle_ticket` | 447 | 46 | 1,341 | 690 | **2,031** |

**Output was 82% of the baseline's cost** (7,245 of 8,814), so that is where the main saving came from.

### What we changed and why

1. **Rewrote the instruction so it no longer triggers verbose output.**
   - The original asked the model to *"Think step by step. Write full reasoning, quote any article…"*.
   - In `shared/llm.py` the client switches to a ≥480-token reasoning dump whenever the prompt contains phrases such as "step by step" or "full reasoning". Output tokens cost 15 points each, 5× an input token.
   - The new instruction asks only for the JSON object (`action`, `kb_article`, `customer_reply`), lists the four allowed actions, and contains none of the trigger phrases.
   - Effect: output fell from 483 to 46 tokens on T-1001.
   - The product only uses the JSON, so the reasoning text was pure cost.
2. **Removed `Run id {uuid4()}` from the system prompt.** It carried no information for the model, added tokens to every call, and would make every prompt unique, which rules out any provider-side prefix caching.
3. **Trimmed the few-shot examples instead of deleting them.**
   - All four examples are kept as one compact line each (situation → action, article), not full JSON lines with headers.
   - They still anchor the decision pattern and the exact article IDs, for fewer tokens.
4. **Order snapshot is serialized as compact JSON** (`separators=(",", ":")` instead of `indent=2`). Indentation whitespace costs tokens and adds no meaning.
5. **Dropped the `TICKET_ID:` line.** An internal ID doesn't affect the decision.
6. **Put the help center before the few-shot and the instruction**, so the instruction is the last thing the model reads before the ticket.

Changes 2–6 together reduced input from 523 to 447 tokens (T-1001).

---

## Question 2: four-node pipeline

### Per-ticket results

| Ticket | Baseline points | Ours: LLM calls | Ours: points |
|---|---:|---:|---:|
| T-1001 | 73,599 | 1 | **1,728** |
| T-1002 | 73,494 | 1 | **1,605** |
| T-1003 | 73,608 | 1 | **1,566** |
| T-1004 | 73,449 | 1 | **1,527** |
| T-1091 (holdout) | 73,656 | 1 | **1,569** |
| **Public total** | **294,150** | | **6,426** |

### Points breakdown: physical fulfilment path, T-1004 (REPLACE)

| Node | Baseline in / out tok | Baseline pts | Ours in / out tok | Ours pts |
|---|---:|---:|---:|---:|
| triage | 8,438 / 493 | 32,709 | code: no LLM call | 0 |
| retrieve | 603 / 36 | 2,349 | code: no LLM call | 0 |
| compose | 1,176 / 492 | 10,908 | 329 / 36 | **1,527** |
| audit | 6,701 / 492 | 27,483 | code: no LLM call | 0 |
| **Total** | | **73,449** | | **1,527** |

### Points breakdown: escalation path, T-1003 (ESCALATE_TIER2)

| Node | Baseline in / out tok | Baseline pts | Ours in / out tok | Ours pts |
|---|---:|---:|---:|---:|
| triage | 8,437 / 494 | 32,721 | code: no LLM call | 0 |
| retrieve | 604 / 39 | 2,397 | code: no LLM call | 0 |
| compose | 1,179 / 495 | 10,962 | 327 / 39 | **1,566** |
| audit | 6,706 / 494 | 27,528 | code: no LLM call | 0 |
| **Total** | | **73,608** | | **1,566** |

### The deciding constraint
- Even without verbose output, each LLM call returns 36–46 output tokens, about **540–690 points per call in output alone**.
- Four LLM calls would spend almost the whole 2,500 budget on output before counting any input.
- So the pipeline keeps all four stages, but only one of them, compose, calls the LLM.

### What we changed and why

1. **Triage became deterministic retrieval (no LLM call).**
   - The baseline pasted 120,000 characters of the help center (about 8,400 tokens) into an LLM prompt that also contained "step by step". It cost about 32,700 points per ticket.
   - Triage now scores each help-center article against the customer thread plus the order's facts: string values and the names of flags that are true, such as `damage_reported` or `customized`.
   - Scoring is word overlap, with words that appear in many articles counting less (IDF weighting). The top 3 articles are kept.
   - The correct article was in the top-3 shortlist for all 5 tickets.
   - It only shortlists; the decision is still the LLM's, in compose.
2. **Retrieve became a lookup (no LLM call).** Asking an LLM to "print full text" of articles costs tokens just to copy text the code already has. `load_article()` returns it for free.
3. **Compose is the single LLM call.**
   - Its prompt holds: the 3 shortlisted articles, the same trimmed few-shot and trigger-free instruction as Q1 (imported from `question1/agent.py` so both tracks stay in sync), the thread, and the compact order snapshot.
   - The baseline compose prompt also contained the triage output **twice** ("TRIAGE" and "TRIAGE AGAIN") plus the retrieve output. All of that is gone.
4. **Audit became output validation (no LLM call).**
   - The baseline re-sent 80,000 characters of the help center plus all previous outputs, with "double-check" and "restate ticket" (both verbose triggers), at about 27,500 points.
   - Now audit checks that the JSON parses, that `action` is one of the four allowed values, and that `kb_article` is a real article ID.
   - Anything invalid goes to `ESCALATE_TIER2`, the safe human handoff. No public ticket reaches this fallback.
5. **Removed the `trace-{uuid4()}` line.** It lived in the triage prompt, which no longer exists.
6. **Stopped building the 210k-token help-center blob in `__init__`.** The pipeline no longer needs it. In our measurement, building it took about 155 seconds per `SupportPipeline()`, because `expanded_help_center_blob()` re-counts tokens on the whole growing text every loop.

---

## What we deliberately did not touch, and why

- **`shared/`, `eval/` and `data/`.** The billing rules, simulator, tokenizer and ground truth define the problem. Changing them would change the measurement, not the agent.
- **The order snapshot's fields.** Every field is still sent; only the whitespace was removed. The decision depends on these fields: the holdout T-1091 hinges on `subscription_age_days` (38 days, so no refund). Trimming fields to save a few tokens would risk wrong answers on unseen tickets.
- **The help-center text in Q1.** All five articles stay in the prompt. They are small, and a real model needs the policy text to handle tickets the few-shot doesn't cover (for example the subscription holdout).
- **`customer_reply` and `parse_resolution`.**
  - The reply is part of the product.
  - In this simulator, non-verbose output always has the same shape, so removing the field from the instruction would save nothing here.
  - With a real API we would also cap output with `max_tokens` / structured output.
- **`eval/reference.py`.** Never imported by either agent, as the README requires. There is no hardcoded decision table either: decisions come from the model reading the policy articles.
- **Tests.**
  - The correctness tests pass unchanged.
  - The two `test_baseline_cost_is_high` tests (asserting > 15,000 and > 70,000) now fail, **because cost dropped**. They describe the old baseline and were left as they are.
  - The current suite result is 4 passed, 2 failed (those two).

### Why we did not use prompt caching

Caching was considered and rejected on purpose.

- **It cannot apply as built.** The cache needs an identical prompt prefix of ≥ 1,024 tokens.
  - Q1's whole prompt is about 430–450 tokens.
  - Q2's compose prompt is 237 tokens of system text + 92 of ticket data (T-1004), and its system text varies per ticket because it contains the shortlisted articles.
- **It could pay off by padding, but only at volume.** A cached 1,024-token prefix costs 1,024 × 0.3 ≈ 307 points, the same as about 102 uncached tokens. So padding a fixed prefix (all articles + instructions) up to 1,024 tokens and caching it would be cheaper *once the cache is warm*. Projected from the billing formula for T-1004 (not measured):
  - warm call: 307 (cached prefix) + 276 (ticket data) + 540 (output) ≈ **1,123 points**, versus 1,527 now;
  - cold first call: the prefix is billed uncached, 3,072 + 276 + 540 ≈ **3,888 points**, which is over the 2,500 target.
- **Break-even.** About 6 warm calls are needed to recover each cold call: (3,888 − 1,527) / (1,527 − 1,123) ≈ 5.8.
  - Over the four public tickets (1 cold + 3 warm, using T-1004's figures for each), caching would average about 1,814 points, worse than the 1,607 we have.
  - The local simulator credits a cache hit even on the first call. It never checks that the prefix was cached earlier. Counting that as a saving would overstate the result.
- **When to revisit.** At steady production traffic, or with a help center whose real policy text alone exceeds 1,024 tokens (no padding needed), a stable cached prefix becomes worth it.

## Limitations and next steps
- **Shortlisting.** Keyword-overlap triage is appropriate for a small help center. For a large one, swap in BM25 or embedding search at the same point in the pipeline.
- **Shortlist coverage.** If the correct article is not in the top 3, compose cannot cite it. A cheap safeguard: if the model's `kb_article` isn't in the shortlist, re-run with a larger shortlist before escalating.
- **Verbose triggers in customer text.** The verbose check scans the whole prompt, including the customer's own words. A customer who writes "step by step" would trigger it, and the prompt can't prevent that. With a real API, an output-token cap covers this case.
