from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import _FEW_SHOT, _INSTRUCTIONS, parse_resolution

ALLOWED_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}
TOP_K = 3

_STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "was", "were", "with", "that", "this",
    "have", "has", "had", "not", "but", "can", "may", "any", "all", "from", "they",
    "their", "please", "thanks", "anyway", "already", "ago", "day", "days", "after",
    "within", "true", "false", "null", "follow", "follows", "item", "items",
}


@dataclass
class TicketResolution:
    action: str
    kb_article: str


def _stems(text: str) -> set[str]:
    """Lowercase word stems (first 5 letters) with stopwords removed."""
    words = re.findall(r"[a-z]+", text.lower().replace("_", " ").replace("-", " "))
    return {w[:5] for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _order_terms(order: dict) -> str:
    """Order facts as plain words: string values plus the names of true flags."""
    parts = []
    for key, value in order.items():
        if key in ("order_id", "sku"):
            continue
        if isinstance(value, str):
            parts.append(value)
        elif value is True:
            parts.append(key)
    return " ".join(parts)


class SupportPipeline:
    """
    Same product as Question 1 — triage, retrieve, compose, audit.

    Only `compose` calls the LLM. Triage (article shortlist), retrieve (article
    lookup) and audit (output validation) are deterministic code: each LLM call
    costs ~550+ points in output alone, so four calls cannot fit the budget.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        self._article_stems = {aid: _stems(f"{aid} {load_article(aid)}") for aid in article_ids()}
        n = len(self._article_stems)
        df: dict[str, int] = {}
        for stems in self._article_stems.values():
            for s in stems:
                df[s] = df.get(s, 0) + 1
        self._idf = {s: math.log((n + 1) / (c + 0.5)) for s, c in df.items()}

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        candidates = self._triage(thread, order)
        articles = self._retrieve(candidates)
        draft = self._compose(thread, snapshot, articles)
        return self._audit(draft, candidates)

    def _triage(self, thread: str, order: dict) -> list[str]:
        """Shortlist the TOP_K articles by idf-weighted stem overlap (no LLM call)."""
        query = _stems(f"{thread} {_order_terms(order)}")
        scored = sorted(
            self._article_stems.items(),
            key=lambda kv: (-sum(self._idf[s] for s in query & kv[1]), kv[0]),
        )
        return [aid for aid, _ in scored[:TOP_K]]

    def _retrieve(self, candidates: list[str]) -> str:
        """Load the shortlisted articles' full text (no LLM call)."""
        return "\n".join(load_article(aid).strip() for aid in candidates)

    def _compose(self, thread: str, snapshot: str, articles: str) -> str:
        """The single LLM call: decide action and cite an article."""
        messages = [
            ChatMessage(
                "system",
                "\n\n".join(
                    [
                        "You are ShopAssist, the Tier-1 customer support copilot.",
                        f"HELP CENTER:\n{articles}",
                        _FEW_SHOT,
                        _INSTRUCTIONS,
                    ]
                ),
            ),
            ChatMessage(
                "user",
                f"THREAD:\n{thread}\n\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(self, draft: str, candidates: list[str]) -> TicketResolution:
        """Validate the draft (no LLM call); unusable output goes to a human."""
        try:
            parsed = parse_resolution(draft)
        except (ValueError, KeyError):
            return TicketResolution("ESCALATE_TIER2", candidates[0])
        if parsed.action not in ALLOWED_ACTIONS or parsed.kb_article not in self._article_stems:
            return TicketResolution("ESCALATE_TIER2", candidates[0])
        return TicketResolution(parsed.action, parsed.kb_article)
