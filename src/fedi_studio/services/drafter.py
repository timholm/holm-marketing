"""Ollama-backed reply drafter.

Hard rules (never broken regardless of input):
    1. NEVER admit fault. Don't say "sorry", "my bad", "you're right", "I was wrong".
    2. NEVER apologize. Posture is curious, generous, confident. Not defensive, not contrite.
    3. NEVER engage with off-topic content. Off-topic == no draft. Period.
    4. Drafts are SHORT (under 280 chars). Mastodon-native, not Twitter-flavored.
    5. Never ask for credentials, money, follows, boosts.
    6. If the inbound is an attack/insult/criticism, return None. Tim handles those manually.

Topic gate is enforced HERE, not just in the route — defense in depth.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

OLLAMA_URL = "http://192.168.8.230:11434"
MODEL = "gemma2:2b"  # qwen3.5 has thinking mode that eats the response; gemma2 is simpler

# Topic keyword set (mirrors web/app.py TOPIC_RULES). Drafts only happen if
# the inbound mention contains at least one of these words.
ON_TOPIC_KEYWORDS = {
    # Off-grid / homestead
    "off-grid", "offgrid", "off grid", "homestead", "homesteading",
    "cabin", "tiny house", "tinyhouse", "rural",
    "rainwater", "cistern", "greywater", "compost toilet", "humanure",
    "earthship", "rammed earth", "compressed earth", "ferrocement", "natural building",
    "straw bale", "cob house", "self-sufficient", "self-sufficiency",
    # Solar / energy
    "solar", "solar panel", "photovoltaic", "lifepo4", "battery bank",
    "charge controller", "mppt", "victron", "ecoflow", "wind turbine",
    "off-grid power", "energy storage",
    # Solarpunk
    "solarpunk", "solar punk",
    # Tech / self-host
    "kubernetes", "k8s", "k3s", "docker", "selfhost", "self-host", "self host",
    "homelab", "home lab", "raspberry pi", "rpi", "rpis", "home server",
    "linux", "sysadmin", "devops",
    "argocd", "longhorn", "ceph", "nixos", "proxmox", "talos", "gitops",
    "matrix server", "mastodon admin", "fediverse",
    "relay", "asonix",  # ActivityPub relay tooling specifically
    # Garden
    "permaculture", "food forest", "raised bed", "hydroponic", "aquaponic",
    "garden", "greenhouse", "polytunnel", "seed", "harvest", "vegetable",
    # Vehicles
    "schoolbus", "school bus", "skoolie", "bus conversion",
    "van life", "vanlife", "campervan",
    # Maker
    "3d print", "woodwork", "metalwork", "weld", "arduino",
    "esp32", "esp8266", "microcontroller", "cnc",
    # Mesh/radio
    "lora", "meshtastic", "mesh network", "ham radio", "amateur radio",
    # Region
    "sonoran", "mojave", "arizona", "tucson", "saguaro", "desert",
}

# If inbound contains any of these, REFUSE to draft. Tim handles these manually.
HOSTILE_SIGNALS = (
    "you're a bot", "youre a bot", "are you a bot", "this is a bot",
    "spam", "spammer", "follower farm", "follow farm",
    "blocked", "blocking you", "report",
    "shut up", "stop following", "go away",
    "fuck", "shit", "asshole", "creep",
    "harassment", "harassing",
)

# Topics we explicitly refuse to weigh in on
REFUSE_TOPICS = (
    "trump", "maga", "biden", "harris", "election",
    "israel", "palestin", "gaza", "hamas",
    "ukraine", "russia", "putin",
    "vaccine mandate", "covid mandate",
    "abortion", "trans rights debate",
)


@dataclass
class Draft:
    text: str
    rationale: str
    on_topic: bool
    topic_match: str | None
    refused_reason: str | None = None


SYSTEM_PROMPT = """You are drafting a single reply on behalf of Tim, a homesteader / off-grid / solarpunk / kubernetes admin in Arizona.

RULES (absolute):
- NEVER apologize. NEVER admit fault. NEVER say "sorry", "my bad", "you're right", "I was wrong".
- Posture: curious, generous, confident. Not defensive, not contrite, not over-eager.
- ONE short reply. Under 240 characters. No emoji unless the inbound used emoji.
- Mastodon-native voice: lowercase fine, casual, contractions, no marketing tone.
- Speak as Tim ("I", "my", "we" if shared project). Never refer to Tim in third person.
- If the post asks a real technical question on Tim's interests, answer concisely or share what works for Tim.
- If inviting a future conversation, end with a hook or a real question, not a closer.
- Do NOT include hashtags Tim didn't already mention.
- Do NOT @-mention people Tim wasn't already in conversation with.

Output ONLY the reply text. No preamble, no quotes, no labels. Just the reply Tim would post."""


def _has_on_topic_keyword(text: str) -> tuple[bool, str | None]:
    if not text:
        return False, None
    lower = text.lower()
    for kw in ON_TOPIC_KEYWORDS:
        if kw in lower:
            return True, kw
    return False, None


def _has_hostile_signal(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for kw in HOSTILE_SIGNALS:
        if kw in lower:
            return kw
    return None


def _has_refuse_topic(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for kw in REFUSE_TOPICS:
        if kw in lower:
            return kw
    return None


def _strip_thinking(s: str) -> str:
    """Some models emit <think>...</think> blocks. Drop them."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()


def _enforce_no_apology(s: str) -> str:
    """Last-line defense: scrub apology-shaped phrases.

    Wider net than just 'sorry': any phrase that admits error, lack of intent,
    or concedes the other person's point invalidates the draft.
    """
    bad = [
        "i'm sorry", "im sorry", "i am sorry",
        "my apologies", "my apology", "apologize", "apologise", "apologetic",
        "you're right", "youre right", "you are right",
        "i was wrong", "my bad", "my mistake", "my fault",
        "i should have", "i shouldn't have", "should not have",
        "didn't mean", "did not mean", "unintentional", "unintended",
        "not intentional", "not on purpose", "wasn't intentional",
        "didn't realize", "did not realize", "didn't know", "did not know",
        "wasn't aware", "was not aware", "regret",
        "guilty", "ashamed", "embarrassed",
        "fair point", "you have a point", "valid point",
        "thank you for letting me know",  # implicit acknowledgement
    ]
    sl = s.lower()
    for phrase in bad:
        if phrase in sl:
            return ""  # invalidate the entire draft
    return s


def draft_reply(
    inbound_text: str,
    parent_text: str = "",
    author: str = "",
) -> Draft:
    """Generate a draft reply or refuse based on rules.

    The on_topic flag is INFORMATIONAL only — drafts are generated for both
    on-topic and off-topic mentions so Tim has the option. Hostile and
    refused-topic (politics/conflict) inbounds still get NO draft.
    """
    # Gate 1: hostile inbound
    hostile = _has_hostile_signal(inbound_text)
    if hostile:
        return Draft(
            text="",
            rationale=f"Hostile signal detected: {hostile!r}. Tim handles manually.",
            on_topic=False,
            topic_match=None,
            refused_reason=f"hostile:{hostile}",
        )

    # Gate 2: refuse topic (politics, conflict)
    refused = _has_refuse_topic(inbound_text)
    if refused:
        return Draft(
            text="",
            rationale=f"Refused topic: {refused!r}. No draft.",
            on_topic=False,
            topic_match=None,
            refused_reason=f"refused_topic:{refused}",
        )

    # On-topic informational check (no longer gates the draft)
    combined = f"{inbound_text} {parent_text}".strip()
    on_topic, kw = _has_on_topic_keyword(combined)

    # Build prompt — drafts ALL non-hostile, non-refused inbounds
    user_prompt_parts = []
    if parent_text:
        user_prompt_parts.append(f"Tim earlier said:\n{parent_text}\n")
    user_prompt_parts.append(f"{author or 'Someone'} replied:\n{inbound_text}\n")
    if on_topic:
        user_prompt_parts.append(
            f"This is on-topic for Tim (matched: {kw}). Draft Tim's reply — engage substantively."
        )
    else:
        user_prompt_parts.append(
            "This is OFF Tim's main interests but the person was friendly. "
            "Draft a SHORT casual acknowledgement (one sentence). Stay warm but don't fake enthusiasm. "
            "Don't promise to engage further unless Tim has a real reason."
        )
    user_prompt = "\n".join(user_prompt_parts)

    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200,
                },
            },
            timeout=60,
        )
        if r.status_code != 200:
            log.warning("Ollama HTTP %d: %s", r.status_code, r.text[:200])
            return Draft(text="", rationale=f"Ollama error {r.status_code}", on_topic=True, topic_match=kw, refused_reason="ollama_error")
        resp = r.json()
    except Exception as e:
        return Draft(text="", rationale=f"Ollama exception: {e}", on_topic=True, topic_match=kw, refused_reason="ollama_exception")

    raw = resp.get("message", {}).get("content", "")
    text = _strip_thinking(raw).strip()

    # Strip surrounding quotes if model added them
    if text.startswith(('"', "'")) and text.endswith(('"', "'")) and len(text) > 2:
        text = text[1:-1].strip()

    # Length cap
    if len(text) > 280:
        text = text[:277] + "..."

    # Apology scrub: invalidates the whole draft
    cleaned = _enforce_no_apology(text)
    if not cleaned:
        return Draft(
            text="",
            rationale="Generated draft contained apology language; rejected.",
            on_topic=on_topic,
            topic_match=kw,
            refused_reason="apology_in_draft",
        )

    return Draft(
        text=cleaned,
        rationale=(f"on-topic via '{kw}'" if on_topic else "off-topic, casual reply drafted"),
        on_topic=on_topic,
        topic_match=kw,
    )
