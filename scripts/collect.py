"""Collect candidate F1 comments into data/pool.csv for hand-labeling."""
import csv
import os
import re

WS = re.compile(r"\s+")


def clean_text(text):
    return WS.sub(" ", text or "").strip()


def is_filterable(text):
    """True = drop before a human ever sees it (junk, not a label decision)."""
    if not text:
        return True
    if text.lower() in ("[deleted]", "[removed]"):
        return True
    if len(text) < 15:
        return True
    if text.startswith("http") and " " not in text:
        return True
    return False


# Pull each label from where it's densest (see planning.md §4).
# (subreddit, time_filter, num_submissions, comments_per_submission)
SUBREDDIT_SOURCES = [
    ("F1Technical", "year", 25, 15),   # analysis-dense
    ("formula1", "month", 30, 15),     # hot_take / joke / reaction
]
# Paste a few live race-thread / post-race discussion URLs for reaction-dense comments:
SUBMISSION_URLS = [
    # "https://www.reddit.com/r/formula1/comments/XXXXXX/...",
]


def collect(out_path="data/pool.csv"):
    import praw
    from dotenv import load_dotenv
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    seen, rows = set(), []

    def add(c, sub):
        body = clean_text(getattr(c, "body", ""))
        if c.id in seen or is_filterable(body):
            return
        seen.add(c.id)
        rows.append({"id": c.id, "text": body, "source_sub": sub,
                     "score": c.score, "permalink": "https://reddit.com" + c.permalink})

    for sub, tf, nsub, ncom in SUBREDDIT_SOURCES:
        for sm in reddit.subreddit(sub).top(time_filter=tf, limit=nsub):
            sm.comments.replace_more(limit=0)
            for c in sm.comments[:ncom]:
                add(c, sub)
    for url in SUBMISSION_URLS:
        sm = reddit.submission(url=url)
        sm.comments.replace_more(limit=0)
        for c in sm.comments[:50]:
            add(c, sm.subreddit.display_name)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "source_sub", "score", "permalink"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} comments to {out_path}")


if __name__ == "__main__":
    collect()
