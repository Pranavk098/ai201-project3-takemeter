# Groq zero-shot baseline prompt

Paste the template below into the notebook's baseline cell as a Python f-string on `{text}`.
Call `llama-3.3-70b-versatile` with `temperature=0`. Then normalize each response
(lowercase, strip, keep the first exact label match; otherwise record `"unknown"` → counts as wrong).

```text
You are classifying the REGISTER of a comment from Formula 1 fan discussion into exactly one of four labels. Reply with ONLY the label in lowercase, nothing else.

Labels:
- analysis: a structured argument backed by specific, verifiable evidence (lap times, tire strategy, technical/regulation reasoning, statistics). The reasoning is the point.
- hot_take: a bold, confident opinion or judgment asserted without genuine evidence (rankings, overrated/underrated, predictions stated as fact). Any evidence is vague or decorative.
- reaction: an immediate emotional response to a specific event, with little or no argument.
- joke: a comment whose primary intent is humor — meme, pun, bit, sarcasm, copypasta.

Rules:
- If an emotional or ranty tone wraps a real, evidenced argument, it is still analysis (substance over tone).
- If the comedic construction is clearly the point, it is joke; if it is sincere emotion that merely happens to be hyperbolic, use the sincere label.

Comment:
"""{text}"""

Label:
```
