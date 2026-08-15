"""Generate the complete market assessment table (Appendix B) from the CSV.

Mirrors the scoring in `make_charts.py`: 23 scored coverage criteria, four
stages each for the Python SDK and the CLI, four plus documented server hosting
for the MCP, four plus three capabilities for ML platform agents, and the
meta-harness as a criterion of its own. Rows are ordered by market coverage.

    python make_table.py        # rewrites market_assessment.tex
"""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
SRC = HERE / "MLPlatformAgentBench market assessment.csv"
OUT = HERE / "market_assessment.tex"

# (existence column that carries the documenting URL, value columns)
GROUPS = [
    (5, [6, 7, 8, 9]),          # Python SDK
    (10, [11, 12, 13, 14]),     # CLI
    (15, [16, 17, 18, 19]),     # MCP stages
    (20, [20]),                 # MCP server hosting (links to its own cell)
    (21, [22, 23, 24, 25]),     # ML platform agent stages
    (None, [26, 27, 28]),       # agent capabilities: PySpark, file system, dashboards
    (30, [30]),                 # meta-harness (links to its own cell)
    (31, [31]),                 # published skills
    (29, [29]),                 # documented agent deployments
]
NAME, URL = 0, 1
SCORED = [c for _, cols in GROUPS for c in cols]
OTHER_AGENT = 32  # unscored: an agent that does not meet the ML platform agent definition


def is_url(v):
    return v.strip().lower().startswith("http")


def pts(v):
    v = v.strip().lower()
    if is_url(v) or v == "yes":
        return 1.0
    if v == "partial":
        return 0.5
    return 0.0


def first_url(cell):
    """Some cells list several sources; link to the first one."""
    for tok in (cell or "").split():
        if tok.lower().startswith("http"):
            return tok
    return None


def mark(v, href):
    """Bullet / open circle / dash, linked to its documenting source."""
    p = pts(v)
    sym = {1.0: r"$\bullet$", 0.5: r"$\circ$"}.get(p, "--")
    url = first_url(href) if href else None
    if p and url:
        return r"\href{%s}{%s}" % (url.replace("%", r"\%"), sym)
    return sym


def escape(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


rows = list(csv.reader(open(SRC, encoding="utf-8-sig")))[1:]
rows = [r for r in rows if r and r[NAME].strip()]
order = {r[NAME]: i for i, r in enumerate(rows)}
rows.sort(key=lambda r: (-sum(pts(r[c]) for c in SCORED), order[r[NAME]]))

body = []
for r in rows:
    cells = []
    for href_col, cols in GROUPS:
        href = r[href_col] if href_col is not None else None
        cells += [mark(r[c], href) for c in cols]
    # unscored trailing column: the agent the vendor ships that is not an ML
    # platform agent, kept so the judgement is documented rather than dropped
    oa = first_url(r[OTHER_AGENT]) if len(r) > OTHER_AGENT else None
    cells.append(r"\href{%s}{$\dagger$}" % oa.replace("%", r"\%") if oa else "--")
    name = escape(" ".join(r[NAME].split()))
    link = first_url(r[URL])
    name = r"\href{%s}{%s}" % (link, name) if link else name
    body.append(r"\raggedright %s & %s \\" % (name, " & ".join(cells)))

HEAD = r""" & \multicolumn{4}{c}{\glsfmtshort{SDK}} & \multicolumn{4}{c}{\glsfmtshort{CLI}} & \multicolumn{5}{c}{\glsfmtshort{MCP}} & \multicolumn{7}{c}{\glsfmtshort{ML} platform agents} & \multicolumn{1}{c}{MH} & \multicolumn{1}{c}{Sk} & \multicolumn{1}{c}{Ad} & \multicolumn{1}{c}{OA} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-14}\cmidrule(lr){15-21}\cmidrule(lr){22-22}\cmidrule(lr){23-23}\cmidrule(lr){24-24}\cmidrule(lr){25-25}
Platform & F & T & I & O & F & T & I & O & F & T & I & O & H & F & T & I & O & Py & Fs & Db & M & S & A & O \\
\midrule"""

CAPTION = (
    r"\caption[Complete market assessment]{The 23 scored coverage criteria of the 39 "
    r"assessed platforms, ordered by market coverage. F, T, I, and O denote the feature, "
    r"training, inference, and operations stages, H denotes documented \glsfmtshort{MCP} "
    r"server hosting, Py, Fs, and Db denote the PySpark, file-system, and dashboard "
    r"capabilities available through \gls{ML} platform agents, M denotes a documented "
    r"meta-harness, S a published set of skills, and A documented agent deployments. "
    r"A filled circle marks full support, an open circle partial support, "
    r"and a dash no documented support. The trailing column O is not scored, and a dagger there records "
    r"an other software agent, namely a software agent the vendor ships that is not an "
    r"\gls{ML} platform agent, such as one for analytics or office work, an assistant "
    r"that suggests rather than executes, "
    r"or a pipeline agent confined to another domain. In the digital version, each "
    r"platform name links to the platform website and each mark links to the documenting "
    r"source for its interaction class, where one is recorded.}"
)

tex = "\n".join([
    "% Auto generated from docs/thesis/img/market/MLPlatformAgentBench market assessment.csv",
    "% Regenerate with: python docs/thesis/img/market/make_table.py",
    r"{\scriptsize\setlength{\tabcolsep}{2.4pt}",
    r"\begin{longtable}{@{}p{3.1cm} *{24}{c} @{}}",
    CAPTION,
    r"\label{tab:market-coverage-full} \\",
    r"\toprule",
    HEAD,
    r"\endfirsthead",
    r"\caption[]{Scored coverage criteria (continued).} \\",
    r"\toprule",
    HEAD,
    r"\endhead",
    *body,
    r"\bottomrule",
    r"\end{longtable}}",
    "",
])
OUT.write_text(tex, encoding="utf-8", newline="\n")
print("wrote %s (%d platforms, %d scored criteria)" % (OUT.name, len(rows), len(SCORED)))
print("top of table: " + ", ".join(r[NAME] for r in rows[:6]))
