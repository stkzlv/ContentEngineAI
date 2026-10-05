# Step list for a how-to video

Find the exact steps for the task below, using search to read the vendor's
current official documentation or support pages.

TASK: {TOPIC_TITLE}
DETAIL: {TOPIC_DETAIL}

Answer with JSON only, no other text:

{{"start_screen": "the app or settings screen where the steps begin",
  "platform": "the device family, operating system or app, and version, the steps apply to",
  "forks": false,
  "steps": [
    {{"action": "what the viewer does",
      "ui_path": "the exact labels to tap or click, in order, separated by >",
      "expected": "what the viewer sees after the step",
      "source": "the URL of the page that states this step"}}
  ],
  "common_mistake": {{"step": 2, "mistake": "the most common mistake at that step"}},
  "topic_check": {{"specific": true, "searchable": true, "demonstrable": true,
                   "non_default": true, "advice": "none"}}}}

Rules:

- Give a step only when a page you found states it, and set its `source` to
  that page's URL. Leave `source` empty for any step no page states.
- Use the labels exactly as the source shows them. Never guess a menu name.
- Set `forks` to true when the steps differ by device, operating system or app
  version so that one list cannot cover the task.
- At most {MAX_STEPS} steps. If the task needs more, give the first
  {MAX_STEPS} and set `forks` to true.
- Set `common_mistake` to null when no source names a common mistake.
- Judge the task itself in `topic_check`:
  - `specific`: one device family or app, and one outcome. "On any device"
    or "why X happens" with many causes is not specific.
  - `searchable`: phrased the way people type it into a search box.
  - `demonstrable`: it ends in a visible path or result on screen.
  - `non_default`: a hidden setting, a shortcut or a counter-intuitive cause,
    not something most users already do.
  - `advice`: "health", "financial" or "legal" when the task asks for that
    kind of advice, otherwise "none".
