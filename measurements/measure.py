"""Measure per-call tokens/points for the original baseline agents vs the optimized ones.

Usage (from the repo root):  python measurements/measure.py > measurements/results.json
Takes ~3 minutes: the baseline Q2 pipeline builds the 210k-token help center.
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).parent))

from eval.reference import TICKET_EXPECTATIONS as E  # measurement only, not used by agents

import baseline_q1, baseline_q2
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004", "T-1091"]
out = {}
for name, cls in [("q1_base", baseline_q1.SupportCopilot), ("q1_new", SupportCopilot),
                  ("q2_base", baseline_q2.SupportPipeline), ("q2_new", SupportPipeline)]:
    agent = cls()
    rows = []
    for t in TICKETS:
        n = len(agent.session.calls)
        r = agent.handle_ticket(t)
        calls = [
            {"label": c.label, "in": c.input_tokens, "cached": min(c.cached_prefix_tokens, c.input_tokens),
             "out": c.output_tokens, "points": c.points()}
            for c in agent.session.calls[n:]
        ]
        rows.append({"ticket": t, "action": r.action, "kb_article": r.kb_article,
                     "correct": (r.action, r.kb_article) == (E[t].action, E[t].kb_article),
                     "calls": calls, "points": sum(c["points"] for c in calls)})
    out[name] = rows
    print(name, "done", file=sys.stderr)
print(json.dumps(out, indent=1))
