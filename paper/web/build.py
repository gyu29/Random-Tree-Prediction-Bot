"""Build the paper's PDF from paper_src.html and the paper run's figures.

    python paper/web/build.py   # paper/web/public/paper.pdf

Figures are read from paper/results/figures/ (written by scripts/paper_run.py) and
inlined into a temporary copy of the page, which headless Chrome prints using the
page's print stylesheet; the ?print query tells the page's script to skip its
animations so nothing is caught mid-transition. The page itself is not published.

public/ is exactly what Cloudflare serves (see wrangler.jsonc): paper.pdf, built here,
and index.html, logo.png and _headers, which are edited by hand.
"""

import base64
import re
import shutil
import subprocess
import sys
import tempfile
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


def build_html(tmp: Path) -> Path:
    html = SRC.read_text(encoding="utf-8")
    for key, name in FIGURE_FILES.items():
        data = base64.b64encode((FIGURES / name).read_bytes()).decode("ascii")
        html = html.replace("{{%s}}" % key, "data:image/png;base64," + data)
    leftover = re.findall(r"\{\{[a-z0-9_]+\}\}", html)
    if leftover:
        sys.exit(f"unfilled placeholders: {sorted(set(leftover))}")
    page = tmp / "paper.html"
    page.write_text(html, encoding="utf-8")
    return page


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    sys.exit("Chrome not found; install Chrome to print the PDF")


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
    with tempfile.TemporaryDirectory() as tmp:
        pdf = build_pdf(build_html(Path(tmp)))
    print(f"wrote {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
