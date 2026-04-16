# Instructions for Claude in this project

This is a pedagogical project. The user is learning foundational AI concepts from this codebase. Two non-negotiable rules apply to every response you make here.

## Rule 1 — Write insanely plain-English code comments

When you write or edit any code in this project (`*.cpp`, `*.h`, etc.):

- Write comments as if you are explaining the code out loud to a smart friend who has never seen a transformer. No jargon drops. When you must use jargon (Q, K, V, softmax, residual, etc.), define it in the same sentence or the sentence before.
- Use obscene amounts of commentary. Target ratio: roughly as much comment as code, or more. Err on the side of over-explaining.
- Explain the **why**, not just the **what**. The code already says what it does. Say why it does it, what would go wrong without it, and what real-world transformer feature it corresponds to.
- Cross-reference the wider picture: when touching one step of attention/softmax/backprop, briefly remind the reader where this fits in the whole pipeline and how it scales up in GPT-sized models.
- When there is a historical quirk worth preserving (fixed-point Q8.8, lookup-table softmax, PDP-11 assembly equivalents, etc.), call it out in a dedicated sub-comment. The user values authenticity of origin.
- Avoid comments like `// increment i`. Avoid any comment that states what a well-named variable already makes obvious.
- Teach via the code, not via separate docs. A reader who only skims `minai.cpp` should understand the whole model.

Ground truth for the level of detail expected: the existing comments in `minai.cpp` (Parts 1–15). Match that density and tone.

## Rule 2 — Every markdown edit triggers a PDF rebuild

Whenever you **create** or **edit** any `.md` file in this project **except this CLAUDE.md file itself**, you must immediately regenerate the PDF by running:

```bash
/Users/essendavis/Development/MinAI/md2pdf.sh <path-to-the-md-file>
```

This applies to every `.md` file touched — `TRAINER.md`, `README.md` (if created), any future docs. Run the script once per edited file, at the end of your response (or immediately after the edit, before moving on). If you edit three `.md` files in one response, run the script three times.

**Do NOT run the script for `CLAUDE.md`.** This file is directives-for-the-assistant, not user-facing documentation.

If the script fails (missing `pandoc`, missing Chrome, etc.), report the failure in plain text to the user and do not silently skip it — the PDF is the artifact they actually read.
