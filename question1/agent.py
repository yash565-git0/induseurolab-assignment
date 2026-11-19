from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import all_articles_blob
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


_FEW_SHOT = """Examples:
unopened item, day 10: REFUND, returns-within-30-days
ebook, license shown: SEND_ARTICLE, digital-license-no-refund
cracked custom mug: ESCALATE_TIER2, custom-items-no-replace
cracked stock mug, day 2: REPLACE, damage-replacement-7-days"""

_INSTRUCTIONS = (
    "Respond with only a JSON object with keys action, kb_article, customer_reply. "
    "action is one of REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2. "
    "kb_article is the id of the help-center article that supports the action."
)


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (baseline).

    Given a support ticket + order snapshot, choose an action, cite the help-center
    article, and draft the customer reply.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        system = "\n\n".join(
            [
                "You are ShopAssist, the Tier-1 customer support copilot.",
                "HELP CENTER:",
                all_articles_blob(),
                _FEW_SHOT,
                _INSTRUCTIONS,
            ]
        )
        user = "\n\n".join(
            [
                f"THREAD:\n{thread}",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ]
        )
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
