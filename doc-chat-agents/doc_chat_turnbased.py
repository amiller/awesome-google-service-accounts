#!/usr/bin/env python3
"""Turn a Google Doc into a turn-based chat with two agents (via bitrouter).

Type a line starting with "Me:" in the doc and save. Haiku replies first,
then CalendarBot (which can actually read/write its own Google Calendar),
then a new "Me:" prompt appears.

Prereqs (see README): `credentials.json` — a service account key shared with
the target doc. `calendar-credentials.json` — a second service account key
(gives CalendarBot its own identity/calendar; reuse the same key here if you
don't care about that distinction). `BITROUTER_API_KEY` env var set.

Usage:
  python3 doc_chat_turnbased.py DOC_ID [--once] [--note "what changed"]
"""
import json
import os
import re
import sys
import time

from google.oauth2 import service_account
from googleapiclient.discovery import build
from anthropic import Anthropic
from openai import OpenAI

DOCS_CREDS = "credentials.json"
CALENDAR_CREDS = "calendar-credentials.json"
BITROUTER_URL = "https://api.bitrouter.ai"
MODEL = "anthropic/claude-haiku-4.5"
USER_TAG = "Me:"
POLL_SECONDS = 1
TYPING = "…typing…"

# bitrouter's Anthropic-model route 502s whenever a `tools` array is attached
# (reproduced on both /v1/messages and /v1/chat/completions) — so CalendarBot
# runs on an OpenAI model instead, where bitrouter's tool-calling works fine.
CALENDAR_MODEL = "openai/gpt-5.4-mini"
CALENDAR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List upcoming events on calendar-bot's own calendar.",
            "parameters": {"type": "object", "properties": {"max_results": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create an event on calendar-bot's own calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 datetime, e.g. 2026-07-03T15:00:00Z"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                },
                "required": ["summary", "start", "end"],
            },
        },
    },
]


def run_calendar_tool(cal, name, args):
    if name == "list_events":
        events = (
            cal.events()
            .list(calendarId="primary", maxResults=args.get("max_results", 10), singleEvents=True, orderBy="startTime")
            .execute()
            .get("items", [])
        )
        return [{"summary": e.get("summary"), "start": e["start"].get("dateTime", e["start"].get("date"))} for e in events]
    if name == "create_event":
        event = (
            cal.events()
            .insert(
                calendarId="primary",
                body={"summary": args["summary"], "start": {"dateTime": args["start"]}, "end": {"dateTime": args["end"]}},
            )
            .execute()
        )
        return {"id": event["id"], "htmlLink": event.get("htmlLink")}
    raise ValueError(f"unknown tool {name}")


def get_docs():
    creds = service_account.Credentials.from_service_account_file(DOCS_CREDS, scopes=["https://www.googleapis.com/auth/documents"])
    return build("docs", "v1", credentials=creds)


def get_calendar():
    creds = service_account.Credentials.from_service_account_file(CALENDAR_CREDS, scopes=["https://www.googleapis.com/auth/calendar"])
    return build("calendar", "v3", credentials=creds)


def get_doc(docs, doc_id):
    doc = docs.documents().get(documentId=doc_id).execute()
    text = "".join(
        run.get("textRun", {}).get("content", "")
        for el in doc["body"]["content"]
        for run in el.get("paragraph", {}).get("elements", [])
    )
    return text, doc["body"]["content"][-1]["endIndex"]


def append_text(docs, doc_id, text):
    # endOfSegmentLocation always means "wherever the doc currently ends" — unlike a
    # cached index, it can't go stale if the doc changed since we last read it.
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"endOfSegmentLocation": {}, "text": text}}]},
    ).execute()
    # Google Docs inherits the preceding run's formatting for new text, so without this
    # a bot's highlight color bleeds forward into the next "Me:" prompt / hotswap note.
    _, end_index = get_doc(docs, doc_id)
    start = end_index - 1 - len(text)
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {
                    "updateTextStyle": {
                        "range": {"startIndex": start, "endIndex": end_index - 1},
                        "textStyle": {"backgroundColor": {"color": {}}},
                        "fields": "backgroundColor",
                    }
                }
            ]
        },
    ).execute()


def insert_typing_indicator(docs, doc_id, tag):
    placeholder = f"{tag} {TYPING}"
    append_text(docs, doc_id, f"\n\n{placeholder}")
    _, end_index = get_doc(docs, doc_id)  # learn where it actually landed
    start = end_index - 1 - len(placeholder)
    return start, len(placeholder)


def replace_typing_indicator(docs, doc_id, start, placeholder_len, tag, reply, highlight):
    tag_and_reply = f"{tag} {reply}"
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {"deleteContentRange": {"range": {"startIndex": start, "endIndex": start + placeholder_len}}},
                {"insertText": {"location": {"index": start}, "text": tag_and_reply}},
                {
                    "updateTextStyle": {
                        "range": {"startIndex": start, "endIndex": start + len(tag_and_reply)},
                        "textStyle": {"backgroundColor": {"color": {"rgbColor": highlight}}},
                        "fields": "backgroundColor",
                    }
                },
            ]
        },
    ).execute()


def last_turn(text, all_tags):
    pattern = "|".join(re.escape(t) for t in all_tags)
    parts = re.split(rf"(?m)^({pattern})\s*", text)
    if len(parts) < 3:
        return None, ""
    return parts[-2], parts[-1].strip()


def messages_for(text, all_tags, bot_tag):
    """Build an alternating message list from bot_tag's point of view — everything
    said by anyone else (human or other bots) counts as 'user' input to it. Other
    speakers' tags are kept inline so the model can tell who said what."""
    pattern = "|".join(re.escape(t) for t in all_tags)
    parts = re.split(rf"(?m)^({pattern})\s*", text)
    turns = []
    for tag, content in zip(parts[1::2], parts[2::2]):
        content = content.strip()
        if not content:
            continue
        role = "assistant" if tag == bot_tag else "user"
        labeled = content if role == "assistant" else f"{tag} {content}"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + labeled
        else:
            turns.append({"role": role, "content": labeled})
    return turns


def system_prompt(bot, bots):
    others = ", ".join(
        f"{b['tag']} ({'tools: ' + ', '.join(t['function']['name'] for t in b['tools']) if b.get('tools') else 'no tools, plain chat'})"
        for b in bots
        if b is not bot
    )
    return (
        f"You are {bot['tag'].rstrip(':')}, one of several agents replying in turn in a shared Google Doc chat "
        f"alongside a human (tagged '{USER_TAG}'). Other agents present: {others}. Every message below is prefixed "
        "with who said it — use those tags to track who you're responding to and what other agents have already "
        f"said or done, including their tool results. The doc already labels your own turn with '{bot['tag']}' before "
        "you write anything, so start your reply with your actual message — never repeat a 'Name:' style prefix "
        "(your own or anyone else's) at the start of your text."
    )


def clean_reply(reply, bot_tag, all_tags):
    """Bots sometimes echo a speaker tag anyway — imitating the transcript format
    they see in context — either glued to the very start of their own reply, or
    drifting into narrating another speaker's turn partway through. Strip both,
    client-side: bitrouter's own stop-sequence support is broken on the OpenAI
    route (502s even with no tools attached), so the server can't do this for us."""
    pattern = "|".join(re.escape(t) for t in all_tags)
    while True:
        stripped = re.sub(rf"^({pattern})\s*", "", reply)
        if stripped == reply:
            break
        reply = stripped
    for tag in all_tags:
        if tag == bot_tag:
            continue
        idx = reply.find(f"\n{tag}")
        if idx != -1:
            reply = reply[:idx]
    return reply.strip()


def run_anthropic_bot(docs, client, doc_id, text, all_tags, bot):
    messages = messages_for(text, all_tags, bot["tag"])
    start, placeholder_len = insert_typing_indicator(docs, doc_id, bot["tag"])
    resp = client.messages.create(model=bot.get("model", MODEL), max_tokens=1024, system=bot["system"], messages=messages)
    reply = "".join(block.text for block in resp.content if block.type == "text")
    reply = clean_reply(reply, bot["tag"], all_tags)
    replace_typing_indicator(docs, doc_id, start, placeholder_len, bot["tag"], reply, bot["highlight"])


def run_openai_bot(docs, client, doc_id, text, all_tags, bot):
    messages = [{"role": "system", "content": bot["system"]}] + messages_for(text, all_tags, bot["tag"])
    start, placeholder_len = insert_typing_indicator(docs, doc_id, bot["tag"])
    kwargs = {"model": bot["model"], "messages": messages, "tools": bot["tools"]}
    msg = client.chat.completions.create(**kwargs).choices[0].message
    while msg.tool_calls:
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result = bot["executor"](tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        msg = client.chat.completions.create(**{**kwargs, "messages": messages}).choices[0].message
    reply = clean_reply(msg.content or "", bot["tag"], all_tags)
    replace_typing_indicator(docs, doc_id, start, placeholder_len, bot["tag"], reply, bot["highlight"])


def step(docs, clients, doc_id, text, all_tags, bots):
    tag, content = last_turn(text, all_tags)
    if tag is None:
        append_text(docs, doc_id, f"\n{USER_TAG} ")
        print(f"seeded doc with '{USER_TAG}' prompt")
        return
    if tag != USER_TAG or not content:
        return
    for bot in bots:
        text, _ = get_doc(docs, doc_id)  # refresh so later bots see earlier bots' replies
        run = run_openai_bot if bot.get("tools") else run_anthropic_bot
        run(docs, clients[bot["backend"]], doc_id, text, all_tags, bot)
        print(f"{bot['tag']} responded")
    append_text(docs, doc_id, f"\n\n{USER_TAG} ")


def main():
    doc_id = sys.argv[1]
    once = "--once" in sys.argv
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else None
    docs = get_docs()
    cal = get_calendar()
    key = os.environ["BITROUTER_API_KEY"]
    clients = {
        "anthropic": Anthropic(api_key=key, base_url=BITROUTER_URL),
        "openai": OpenAI(api_key=key, base_url=f"{BITROUTER_URL}/v1"),
    }
    bots = [
        {"tag": "Haiku:", "backend": "anthropic", "highlight": {"red": 1, "green": 0.93, "blue": 0.55}},  # soft yellow
        {
            "tag": "CalendarBot:",
            "backend": "openai",
            "model": CALENDAR_MODEL,
            "highlight": {"red": 0.75, "green": 0.95, "blue": 0.8},  # soft green
            "tools": CALENDAR_TOOLS,
            "executor": lambda name, args: run_calendar_tool(cal, name, args),
        },
    ]
    for b in bots:
        b["system"] = system_prompt(b, bots)
    all_tags = [USER_TAG] + [b["tag"] for b in bots]
    if note:
        append_text(docs, doc_id, f"\n[hotswap: {note}]\n")
    print(f"Watching doc {doc_id} — write under '{USER_TAG}', bots reply: {', '.join(b['tag'] for b in bots)}")
    last_text = None
    while True:
        text, _ = get_doc(docs, doc_id)
        # only act once the doc has stopped changing between polls — the closest a
        # polling script gets to "waits until you're done typing" without keystroke events
        if once or text == last_text:
            step(docs, clients, doc_id, text, all_tags, bots)
        last_text = text
        if once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
