"""Build the paper's web page and PDF from paper_src.html and the paper run's figures.

    python paper/web/build.py            # paper/web/public/index.html and paper.pdf
    python paper/web/build.py --no-pdf   # index.html only

Figures are read from paper/results/figures/ (written by scripts/paper_run.py) and
inlined, so index.html is one self-contained file. The PDF is printed from that page
with headless Chrome, using the page's print stylesheet; the ?print query tells the
page's script to skip its animations so nothing is caught mid-transition.

public/ is exactly what Cloudflare serves (see wrangler.jsonc): the two generated
files plus logo.png and _headers, which are edited by hand.
"""

import argparse
import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper" / "web" / "paper_src.html"
FIGURES = ROOT / "paper" / "results" / "figures"
OUT = ROOT / "paper" / "web" / "public"

# Paper figure number -> paper_run.py figure file. The paper orders figures by where
# they are discussed, which is not the order paper_run.py writes them in.
FIGURE_FILES = {
    "fig1": "fig7_effective_series.png",
    "fig2": "fig5_split_leakage.png",
    "fig3": "fig2_standard_error_inflation.png",
    "fig4": "fig4_independent_outcomes_by_horizon.png",
    "fig5": "fig6_test_reliability.png",
    "fig6": "fig1_marginal_return_curves.png",
    "fig7": "fig3_evidence_against_bars.png",
    "fig8": "fig8_ranking_sources.png",
}

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]


def build_html() -> Path:
    html = SRC.read_text(encoding="utf-8")
    for key, name in FIGURE_FILES.items():
        data = base64.b64encode((FIGURES / name).read_bytes()).decode("ascii")
        html = html.replace("{{%s}}" % key, "data:image/png;base64," + data)
    leftover = re.findall(r"\{\{[a-z0-9_]+\}\}", html)
    if leftover:
        sys.exit(f"unfilled placeholders: {sorted(set(leftover))}")
    out = OUT / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    sys.exit("Chrome not found; rerun with --no-pdf or install Chrome")


def build_pdf(page: Path) -> Path:
    pdf = OUT / "paper.pdf"
    subprocess.run(
        [
            find_chrome(),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            page.as_uri() + "?print",
        ],
        check=True,
        capture_output=True,
    )
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdf", action="store_true", help="skip printing the PDF")
    args = parser.parse_args()
    page = build_html()
    print(f"wrote {page.relative_to(ROOT)}")
    if not args.no_pdf:
        print(f"wrote {build_pdf(page).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
