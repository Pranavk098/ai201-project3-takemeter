"""Turn the raw Apify dataset dump into a clean candidate pool (data/pool.csv).

Filters out posts, bot/automod notices, deleted/removed, and too-short comments;
dedupes by id; collapses whitespace; tags each comment's subreddit by the most
recent preceding post row (comment rows don't carry the subreddit).
"""
import csv
import json
import re
import sys

WS = re.compile(r"\s+")
BOT_MARKERS = (
    "i am a bot",
    "performed automatically",
    "contact the moderators",
    "we remind everyone",
    "this is a reminder",
    "your submission has been removed",
)


def clean_text(t):
    return WS.sub(" ", t or "").strip()


def is_junk(t):
    if not t:
        return True
    low = t.lower()
    if low in ("[deleted]", "[removed]"):
        return True
    if any(m in low for m in BOT_MARKERS):
        return True
    if len(t) < 15:
        return True
    if t.startswith("http") and " " not in t:
        return True
    return False


def main(raw_path, out_path="data/pool.csv"):
    with open(raw_path, encoding="utf-8") as f:
        items = json.load(f)["items"]

    seen, rows, cur_sub, cur_post = set(), [], None, None
    for it in items:
        if it.get("type") == "post":
            cur_sub = it.get("subreddit")
            cur_post = it.get("title")
            continue
        body = clean_text(it.get("body"))
        cid = it.get("id")
        if cid in seen or is_junk(body):
            continue
        seen.add(cid)
        rows.append({
            "id": cid,
            "text": body,
            "source_sub": cur_sub,
            "score": it.get("score", 0),
            "post_title": clean_text(it.get("postTitle") or cur_post),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "source_sub", "score", "post_title"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    by_sub = Counter(r["source_sub"] for r in rows)
    print(f"kept {len(rows)} comments -> {out_path}")
    print("by subreddit:", dict(by_sub))
    print("avg length:", round(sum(len(r["text"]) for r in rows) / max(len(rows), 1)))


if __name__ == "__main__":
    main(sys.argv[1])
