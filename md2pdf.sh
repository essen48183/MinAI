#!/bin/bash
# Convert markdown to PDF via pandoc → HTML → Chrome headless
# Usage: ./md2pdf.sh input.md [output.pdf]

set -e

INPUT="${1:?Usage: ./md2pdf.sh input.md [output.pdf]}"
OUTPUT="${2:-${INPUT%.md}.pdf}"

# Put temp HTML next to input so relative image paths resolve correctly
INPUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
TMPHTML="$INPUT_DIR/.md2pdf_tmp.html"

trap "rm -f '$TMPHTML'" EXIT

# Pandoc → standalone HTML with math as images (no JS needed)
pandoc "$INPUT" \
    --standalone \
    --webtex \
    --metadata title="" \
    -c "data:text/css,
body { font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; font-size: 14px; color: %231a1a1a; }
h1 { font-size: 1.8rem; border-bottom: 2px solid %23333; padding-bottom: 0.5rem; }
h2 { font-size: 1.4rem; margin-top: 2.5rem; border-bottom: 1px solid %23ccc; padding-bottom: 0.3rem; }
h3 { font-size: 1.15rem; margin-top: 1.5rem; }
code { background: %23f4f4f4; padding: 0.15em 0.4em; border-radius: 3px; font-size: 0.9em; }
pre { background: %23f4f4f4; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9em; }
th, td { border: 1px solid %23ddd; padding: 0.5rem 0.75rem; text-align: left; }
th { background: %23f0f0f0; font-weight: 600; }
blockquote { border-left: 3px solid %236c8cff; margin: 1rem 0; padding: 0.5rem 1rem; color: %23555; }
hr { border: none; border-top: 1px solid %23ddd; margin: 2rem 0; }
" \
    -o "$TMPHTML"

# Chrome headless → PDF (allow network for webtex math images)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless \
    --disable-gpu \
    --no-pdf-header-footer \
    --allow-file-access-from-files \
    --run-all-compositor-stages-before-draw \
    --virtual-time-budget=5000 \
    --print-to-pdf="$OUTPUT" \
    "file://$TMPHTML" \
    2>/dev/null

echo "$(wc -c < "$OUTPUT" | tr -d ' ') bytes → $OUTPUT"
