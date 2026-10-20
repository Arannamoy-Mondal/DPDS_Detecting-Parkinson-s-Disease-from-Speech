"""Tiny helper: build a short English summary PDF (pdflatex) from verified values passed in by the caller."""
import re
import subprocess
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

SUMMARY_DIR = WORKSPACE / "reports/FINAL_V1/summaries"


def esc(s):
    s = str(s)
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in s)


def table_tex(caption, header, rows, colspec=None):
    n = len(header)
    colspec = colspec or "l" * n
    out = ["\\begin{table}[H]\\centering\\scriptsize\\renewcommand{\\arraystretch}{1.1}", "\\caption{%s}" % esc(caption),
           "\\begin{tabular}{%s}\\toprule" % colspec, " & ".join("\\textbf{%s}" % esc(h) for h in header) + " \\\\ \\midrule"]
    for r in rows:
        out.append(" & ".join(esc(c) for c in r) + " \\\\")
    out += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(out)


def summary_pdf(name, title, subtitle, sections, tables=(), status="COMPLETED"):
    """sections: list of (heading, [bullet strings]); tables: list of (caption, header, rows[, colspec])."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    body = []
    for head, bullets in sections:
        body.append("\\subsection*{%s}" % esc(head))
        body.append("\\begin{itemize}\\itemsep1pt\\parsep0pt" + "".join("\\item %s\n" % esc(b) for b in bullets) + "\\end{itemize}")
    for t in tables:
        body.append(table_tex(*t))
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{booktabs,float,helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[hidelinks]{hyperref}
\pagestyle{plain}
\begin{document}
{\Large\bfseries %s\par}
\vspace{2pt}{\small %s\par}
\vspace{2pt}{\small Protocol FINAL\_V1 \quad Status: \textbf{%s} \quad Generated: %s\par}
\vspace{4pt}\hrule
%s
\end{document}
""" % (esc(title), esc(subtitle), esc(status), esc(now_utc()), "\n".join(body))
    d = safe_path(SUMMARY_DIR / "_build" / name)
    d.mkdir(parents=True, exist_ok=True)
    (d / (name + ".tex")).write_text(tex, encoding="utf8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=d, capture_output=True, text=True)
    if r.returncode != 0 or not (d / (name + ".pdf")).exists():
        print("FATAL: pdflatex failed for", name, r.stdout[-800:]); sys.exit(8)
    out = safe_path(SUMMARY_DIR / (name + ".pdf"))
    shutil.copy2(d / (name + ".pdf"), out)
    shutil.rmtree(d)
    (SUMMARY_DIR / "_build").rmdir() if not any((SUMMARY_DIR / "_build").iterdir()) else None
    return out
