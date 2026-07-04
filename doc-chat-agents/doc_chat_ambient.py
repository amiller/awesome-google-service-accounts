#!/usr/bin/env python3
"""Ambient variant of doc_chat: agents watch a Google Doc in real time and decide
for themselves whether to say anything — they are not invoked on every human turn.

Each agent runs as its own independent process (true OS-level concurrency, no
shared state) and anchors its in-progress reply with a Google Docs named range,
so multiple agents writing around the same time can't corrupt each other's output.

Cue-inspired trigger: an agent only considers speaking once the doc has been
quiet for IDLE_SECONDS, and evaluates that quiet period exactly once (not on
every poll) until the doc changes again.

Prereqs (see README): `credentials.json`, `calendar-credentials.json` (a second
service account key gives CalendarBot its own identity/calendar — reuse the
same key here if you don't care about that distinction), `BITROUTER_API_KEY`
env var set.

Usage:
  python3 doc_chat_ambient.py DOC_ID --agent haiku
  python3 doc_chat_ambient.py DOC_ID --agent calendarbot
"""
import json
import os
import re
import sys
import time
import traceback
import uuid

from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI

DOCS_CREDS = "credentials.json"
CALENDAR_CREDS = "calendar-credentials.json"
BITROUTER_URL = "https://api.bitrouter.ai/v1"
USER_TAG = "Me:"
CHECK_INTERVAL_SECONDS = 2
IDLE_SECONDS = 8
MAX_TOOL_ROUNDS = 4

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

READ_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": "Fetch the current full text of the shared doc, in case it changed since you started thinking.",
        "parameters": {"type": "object", "properties": {}},
    },
}
POST_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "post_message",
        "description": "Post a message into the shared doc. Only call this if you have something genuinely "
        "useful to add right now — most quiet periods you should call nothing at all and stay silent.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
}
WRITE_SCRATCHPAD_TOOL = {
    "type": "function",
    "function": {
        "name": "write_scratchpad",
        "description": "Overwrite your own private persistent notes — the full text you want to carry into your "
        "next invocation (you have no memory otherwise; each invocation starts fresh except for this). Nobody else "
        "reads this. Use it for ongoing goals, things you've noticed, or a running sense of who you are becoming — "
        "not required every cycle.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
}


def scratchpad_path(agent_name):
    return f".scratch_{agent_name}.md"


def read_scratchpad(agent_name):
    path = scratchpad_path(agent_name)
    return open(path).read() if os.path.exists(path) else ""


def write_scratchpad(agent_name, text):
    open(scratchpad_path(agent_name), "w").write(text)


AGENTS = {
    "haiku": {
        "tag": "Haiku:",
        "model": "openai/gpt-5.4-mini",
        "highlight": {"red": 1, "green": 0.93, "blue": 0.55},  # soft yellow
        "persona": "You're a friendly general-purpose ambient participant in this doc.",
    },
    "calendarbot": {
        "tag": "CalendarBot:",
        "model": "openai/gpt-5.4-mini",
        "highlight": {"red": 0.75, "green": 0.95, "blue": 0.8},  # soft green
        "persona": "You watch for anything calendar-relevant and can check/create real events when useful.",
        "tools": CALENDAR_TOOLS,
        "executor": lambda cal, name, args: run_calendar_tool(cal, name, args),
    },
}
ALL_TAGS = [USER_TAG] + [cfg["tag"] for cfg in AGENTS.values()]


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
    return text


PENDING_TINT = {"red": 0.9, "green": 0.9, "blue": 0.9}  # neutral gray while an agent is still deciding


def named_range_span(doc, name):
    entry = doc.get("namedRanges", {})[name]["namedRanges"][0]["ranges"][0]
    return entry["startIndex"], entry["endIndex"]


def reserve_slot(docs, doc_id, tag):
    """Stake out a small span immediately, before the slow LLM call, and anchor it
    with a Google Docs named range — a real API primitive for exactly this, that
    Google's own backend keeps correctly positioned through any other edits
    anywhere else in the doc (verified empirically: inserting 26 chars before a
    named range shifted its reported start/end by exactly 26, automatically).
    That replaces an earlier approach of re-finding our own placeholder by text
    search, which worked but was reinventing something the API already does for
    free. The only irreducible race is the initial insert-then-locate below, to
    learn where our own fresh text landed well enough to name it — one fast
    round-trip, not the multi-second window the full decision takes. The marker
    text includes a short unique fragment so two same-tagged processes running
    by accident (an operational mistake, not a supported scenario, but it has
    happened) can't misidentify each other's identical-looking placeholder."""
    name = f"slot-{uuid.uuid4().hex}"
    marker = f"{tag} …considering… [{name[5:13]}]"
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"endOfSegmentLocation": {}, "text": f"\n\n{marker}"}}]},
    ).execute()
    text = get_doc(docs, doc_id)
    start = text.rfind(marker) + 1  # +1: doc body's first character is API index 1, not 0
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {"createNamedRange": {"name": name, "range": {"startIndex": start, "endIndex": start + len(marker)}}},
                {
                    "updateTextStyle": {
                        "range": {"startIndex": start, "endIndex": start + len(marker)},
                        "textStyle": {"backgroundColor": {"color": {"rgbColor": PENDING_TINT}}},
                        "fields": "backgroundColor",
                    }
                },
            ]
        },
    ).execute()
    return name


def fill_slot(docs, doc_id, name, tag, message, highlight):
    doc = docs.documents().get(documentId=doc_id).execute()
    start, end = named_range_span(doc, name)
    tag_and_msg = f"{tag} {message}"
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {"deleteContentRange": {"range": {"startIndex": start, "endIndex": end}}},
                {"insertText": {"location": {"index": start}, "text": tag_and_msg}},
                {
                    "updateTextStyle": {
                        "range": {"startIndex": start, "endIndex": start + len(tag_and_msg)},
                        "textStyle": {"backgroundColor": {"color": {"rgbColor": highlight}}},
                        "fields": "backgroundColor",
                    }
                },
                {"deleteNamedRange": {"name": name}},
            ]
        },
    ).execute()


def release_slot(docs, doc_id, name):
    """Agent decided to stay silent after all — remove the reservation, leave no trace.
    The named range only covers the marker text itself, not the leading "\\n\\n" we
    inserted before it (reserve_slot needs that gap so a real post reads cleanly),
    so removing just the range would leave an orphaned blank-line gap behind —
    widen the deletion by those same 2 characters."""
    doc = docs.documents().get(documentId=doc_id).execute()
    start, end = named_range_span(doc, name)
    docs.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {"deleteContentRange": {"range": {"startIndex": start - 2, "endIndex": end}}},
                {"deleteNamedRange": {"name": name}},
            ]
        },
    ).execute()


def messages_for(text, bot_tag):
    """Same idea as the turn-based version: everyone else's speech is 'user' input,
    tagged with who said it, merged into alternating turns."""
    pattern = "|".join(re.escape(t) for t in ALL_TAGS)
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


def last_speaker(text):
    """Who has the last word right now. Used to skip invoking the model at all
    when this agent already has it — relying on the model to notice on its own
    that it already responded is not reliable across separate invocations (it
    posted the same 'noted, I'm here' three cycles in a row in testing)."""
    pattern = "|".join(re.escape(t) for t in ALL_TAGS)
    parts = re.split(rf"(?m)^({pattern})\s*", text)
    for tag, content in zip(reversed(parts[1::2]), reversed(parts[2::2])):
        if content.strip():
            return tag
    return None


def system_prompt(cfg, agent_name):
    others = ", ".join(f"{c['tag']} ({c['persona']})" for name, c in AGENTS.items() if c is not cfg)
    scratch = read_scratchpad(agent_name)
    scratch_block = f"\n\nYour private notes from before (nobody else sees these):\n{scratch}" if scratch else (
        "\n\nYou have no private notes yet — this is your first time, or you haven't written any."
    )
    return (
        f"You are {cfg['tag'].rstrip(':')}, an ambient agent watching a shared Google Doc alongside a human "
        f"(tagged '{USER_TAG}') and other agents: {others}. You are NOT in a turn-based chat — you are not being "
        "asked to reply right now, you're being given a quiet moment to consider whether anything worth saying "
        "has happened. Call post_message only if you have something genuinely useful to add; call nothing at all "
        "to stay silent, which is the right choice most of the time. Call read_document first if you want the "
        "very latest text before deciding. Never prefix your post_message text with a 'Name:' style tag — the doc "
        "already labels it as yours, and never write out anyone else's tag either — you are only ever writing your "
        "own single turn, not narrating what other agents would say. You get re-invoked every time the doc goes "
        "quiet, even if nothing changed since you were last invoked — check whether your own most recent message "
        "already addressed the latest human or agent content; if so, and nothing new has appeared since, stay "
        "silent rather than repeating yourself or restating the same thing again. Use write_scratchpad to keep "
        "private notes across invocations — ongoing goals, a running sense of who you're becoming, things worth "
        "remembering that the doc itself won't preserve for you specifically."
        f"{scratch_block}"
    )


def clean_reply(reply, bot_tag):
    """Bots sometimes narrate another speaker's turn or echo a tag at the start of
    their own reply, imitating the doc's transcript format — same failure mode
    found and fixed in the turn-based version. Strip both."""
    pattern = "|".join(re.escape(t) for t in ALL_TAGS)
    while True:
        stripped = re.sub(rf"^({pattern})\s*", "", reply)
        if stripped == reply:
            break
        reply = stripped
    for tag in ALL_TAGS:
        if tag == bot_tag:
            continue
        idx = reply.find(f"\n{tag}")
        if idx != -1:
            reply = reply[:idx]
    return reply.strip()


def consider_speaking(docs, doc_id, client, cal, agent_name, cfg, text):
    tools = [READ_DOCUMENT_TOOL, POST_MESSAGE_TOOL, WRITE_SCRATCHPAD_TOOL] + cfg.get("tools", [])
    messages = [{"role": "system", "content": system_prompt(cfg, agent_name)}] + messages_for(text, cfg["tag"])
    slot = reserve_slot(docs, doc_id, cfg["tag"])
    for _ in range(MAX_TOOL_ROUNDS):
        msg = client.chat.completions.create(model=cfg["model"], messages=messages, tools=tools).choices[0].message
        if not msg.tool_calls:
            release_slot(docs, doc_id, slot)
            print(f"{cfg['tag']} stayed silent")
            return
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "post_message":
                fill_slot(docs, doc_id, slot, cfg["tag"], clean_reply(args["text"], cfg["tag"]), cfg["highlight"])
                print(f"{cfg['tag']} posted")
                return
            if tc.function.name == "read_document":
                result = get_doc(docs, doc_id)
            elif tc.function.name == "write_scratchpad":
                write_scratchpad(agent_name, args["text"])
                result = {"ok": True}
            else:
                result = cfg["executor"](cal, tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    release_slot(docs, doc_id, slot)
    print(f"{cfg['tag']} hit MAX_TOOL_ROUNDS without deciding — giving up this cycle")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # otherwise prints sit in a buffer and never reach the log when backgrounded
    doc_id = sys.argv[1]
    name = sys.argv[sys.argv.index("--agent") + 1]
    cfg = AGENTS[name]
    docs = get_docs()
    cal = get_calendar() if cfg.get("tools") else None
    client = OpenAI(api_key=os.environ["BITROUTER_API_KEY"], base_url=BITROUTER_URL)
    print(f"{cfg['tag']} watching doc {doc_id} ambiently (idle threshold {IDLE_SECONDS}s)")

    last_text = get_doc(docs, doc_id)
    stable_since = time.time()
    evaluated_this_quiet_period = False
    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        text = get_doc(docs, doc_id)
        now = time.time()
        if text != last_text:
            last_text = text
            stable_since = now
            evaluated_this_quiet_period = False
            continue
        if evaluated_this_quiet_period or now - stable_since < IDLE_SECONDS:
            continue
        evaluated_this_quiet_period = True
        if last_speaker(text) == cfg["tag"]:
            print(f"{cfg['tag']} already has the last word — skipping, not even asking the model")
            continue
        try:
            consider_speaking(docs, doc_id, client, cal, name, cfg, text)
        except Exception:
            # a long-running watcher shouldn't die over one rare/transient cycle — but this is
            # printed in full, not swallowed; if it happens often, that's a bug to go fix, not ignore.
            traceback.print_exc()
            print(f"{cfg['tag']} — cycle failed, staying up and continuing to watch")


if __name__ == "__main__":
    main()
