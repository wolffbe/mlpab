"""Generate the RQ3 optimization charts for the MLPAB thesis.

Covers the evaluation phase arms: the six agent optimized CLI variants
(treatments 5 to 10), the optimized skills on the unmodified CLI (11), and
their combinations (12 to 17), all on Hopsworks with Claude Opus, against the
comparison phase baselines of treatment 1. Reads results.csv directly.
Sized for the KTH template text block (130 mm). Fonts: Times New Roman."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / 'results' / 'results.csv'
OUT = HERE

raw = pd.read_csv(SRC)
for c in ['asserts_passed', 'total_asserts', 'local_time_s', 'cost_usd', 'llm_calls', 'n']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
raw['valid'] = raw['valid'].astype(str).str.lower() == 'true'
raw['success'] = raw['success'].astype(str).str.lower() == 'true'
raw['pass_rate'] = raw['asserts_passed'] / raw['total_asserts']
df = raw.sort_values('n').drop_duplicates(
    subset=['config', 'interface', 'skills', 'category', 'task'], keep='last')
df = df[~df['error'].astype(str).str.contains('grader failed to run', na=False)].copy()
df['pr0'] = df.pass_rate.where(df.valid, 0.0)

BASE = '1_hw-full-cli-sdk-skills-opus'
ARMS = ['opt1-batch', 'opt2-session-reuse', 'opt3-compact-json',
        'opt4-idempotent', 'opt5-quiet', 'opt6-stable-output']
NS_CFG = {a: f'{i + 5}_hw-cli-{a}-opus' for i, a in enumerate(ARMS)}
SK_CFG = {a: f'{i + 12}_hw-cli-{a}-skills-opt-opus' for i, a in enumerate(ARMS)}
OPTSK = '11_hw-cli-skills-optimized-opus'


def sel(cfg, iface=None, sk=None):
    m = df.config == cfg
    if iface is not None:
        m &= df.interface == iface
    if sk is not None:
        m &= df.skills == sk
    return df[m]


base_cli = sel(BASE, 'cli', 'none')
base_sdk = sel(BASE, 'sdk', 'none')

TW = 5.12
plt.rcParams.update({
    'pdf.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.edgecolor': '#bdbdbd', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
})
NS_COLOR, SK_COLOR, SDK_COLOR = '#90a4ae', '#455a64', '#1565c0'


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


labels = ['baseline'] + ARMS
ns_frames = [base_cli] + [sel(NS_CFG[a]) for a in ARMS]
sk_frames = [sel(OPTSK)] + [sel(SK_CFG[a]) for a in ARMS]

# fig: pass fraction per arm, without and with the optimized skills
from matplotlib.ticker import PercentFormatter
fig, ax = plt.subplots(figsize=(TW, 3.2))
x = np.arange(len(labels))
w = 0.38
ons = [f.pr0.mean() for f in ns_frames]
osk = [f.pr0.mean() for f in sk_frames]
b1 = ax.bar(x - w / 2, ons, w * 0.94, label='no skills', color=NS_COLOR)
b2 = ax.bar(x + w / 2, osk, w * 0.94, label='+ optimized skills', color=SK_COLOR)
ax.axhline(ons[0], ls='--', lw=0.9, color='#555555')
ax.axhline(base_sdk.pr0.mean(), ls='--', lw=0.9, color=SDK_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6.5)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, 1.14)
ax.set_ylabel('assertion pass fraction')
handles = [b1, b2,
           Line2D([], [], ls='--', lw=0.9, color='#555555', label='baseline CLI, no skills'),
           Line2D([], [], ls='--', lw=0.9, color=SDK_COLOR, label='SDK, no skills')]
ax.legend(handles=handles, frameon=False, ncol=2, loc='upper center',
          bbox_to_anchor=(0.5, 1.22), fontsize=6.5)
for bars, vals in ((b1, ons), (b2, osk)):
    for b, h in zip(bars, vals):
        if pd.notna(h):
            ax.annotate(f'{h:.0%}', (b.get_x() + b.get_width() / 2, h + 0.01),
                        ha='center', va='bottom', fontsize=5.5)
fig.tight_layout()
save(fig, 'optimizations_pass')

# fig: efficiency, one figure per metric
EFF = [('local_time_s', 'local compute time (s)', '{:.0f}', 'time'),
       ('cost_usd', 'cost (USD)', '{:.2f}', 'cost'),
       ('llm_calls', 'LLM turns', '{:.0f}', 'turns')]
for col, title, fmt, tag in EFF:
    fig, ax = plt.subplots(figsize=(TW, 2.9))
    ons = [f[col].mean() for f in ns_frames]
    osk = [f[col].mean() for f in sk_frames]
    b1 = ax.bar(x - w / 2, ons, w * 0.94, label='no skills', color=NS_COLOR)
    b2 = ax.bar(x + w / 2, osk, w * 0.94, label='+ optimized skills', color=SK_COLOR)
    ax.axhline(ons[0], ls='--', lw=0.9, color='#555555')
    ax.axhline(base_sdk[col].mean(), ls='--', lw=0.9, color=SDK_COLOR)
    ax.set_ylabel(title)
    ax.set_ylim(0, max(max(ons), max(osk)) * 1.38)
    for bars, vals in ((b1, ons), (b2, osk)):
        for b, h in zip(bars, vals):
            if pd.notna(h):
                ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=5, rotation=90,
                            color='#333333')
    ax.legend(handles=[b1, b2,
                       Line2D([], [], ls='--', lw=0.9, color='#555555',
                              label='baseline CLI, no skills'),
                       Line2D([], [], ls='--', lw=0.9, color=SDK_COLOR,
                              label='SDK, no skills')],
              frameon=False, ncol=2, fontsize=6, loc='upper center',
              bbox_to_anchor=(0.5, 1.22))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6.5)
    fig.tight_layout()
    save(fig, f'optimizations_efficiency_{tag}')


# fig: planned contrasts of the optimization family
# like for like: bare arms and the optimized skills vs the unmodified CLI
# without skills; each variant + skills combination vs the unmodified CLI with
# the optimized skills, isolating the CLI change at a fixed skill condition
optsk_frame = sel(OPTSK)
rows_cli = ([(f'{a}', base_cli, sel(NS_CFG[a])) for a in ARMS] +
            [('optimized skills', base_cli, optsk_frame)] +
            [(f'{a} + skills', optsk_frame, sel(SK_CFG[a])) for a in ARMS])
rows_sdk = ([(f'{a}', base_sdk, sel(NS_CFG[a])) for a in ARMS] +
            [('optimized skills', base_sdk, optsk_frame)] +
            [(f'{a} + skills', base_sdk, sel(SK_CFG[a])) for a in ARMS])
METRICS = [('pr0', 'pass fraction'), ('cost_usd', 'cost'),
           ('llm_calls', 'turns'), ('local_time_s', 'time')]
SCOLOR = {'pr0': '#1565c0', 'cost_usd': '#ef6c00', 'llm_calls': '#00897b',
          'local_time_s': '#8e24aa'}
SOFF = {'pr0': 0.3, 'cost_usd': 0.1, 'llm_calls': -0.1, 'local_time_s': -0.3}

def wtest2(a, b, col):
    ia = a.set_index(['category', 'task'])[col] if 'category' in a.columns else a[col]
    ib = b.set_index(['category', 'task'])[col] if 'category' in b.columns else b[col]
    common = ia.index.intersection(ib.index)
    x_, y_ = ia.loc[common].astype(float), ib.loc[common].astype(float)
    ok = x_.notna() & y_.notna()
    x_, y_ = x_[ok], y_[ok]
    d = y_ - x_
    if len(d) == 0 or (d != 0).sum() == 0:
        return 1.0, 0.0
    p = sps.wilcoxon(x_, y_, zero_method='wilcox', method='approx').pvalue
    nz = d[d != 0].values
    r = sps.rankdata(abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(p), float(rb)

def holm(ps):
    p = np.array(ps, float)
    adj = np.full_like(p, np.nan)
    order = np.argsort(p)
    mx = 0.0
    for rank, i in enumerate(order):
        mx = max(mx, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = mx
    return adj

report = open(OUT / 'stats_report.txt', 'w')
for rows, refname, tag in [(rows_cli, 'like for like CLI baseline', 'cli'),
                           (rows_sdk, 'SDK without skills', 'sdk')]:
    fig, ax = plt.subplots(figsize=(TW, 4.4))
    res = {}
    for col, lab in METRICS:
        tested = [wtest2(a, b, col) for _, a, b in rows]
        res[col] = list(zip([t[1] for t in tested], holm([t[0] for t in tested])))
        nsig = sum(1 for _, padj in res[col] if padj < 0.05)
        report.write(f'{tag} {lab}: {nsig}/{len(rows)} sig after Holm\n')
        for (name, _, _), (rb, padj) in zip(rows, res[col]):
            mark = '*' if padj < 0.05 else ' '
            report.write(f'   {name:30s} rb={rb:+.2f} holm={padj:.4f}{mark}\n')
    y = np.arange(len(rows))[::-1]
    for col, _ in METRICS:
        for yi, (rb, padj) in zip(y, res[col]):
            filled = padj < 0.05
            ax.scatter(rb, yi + SOFF[col], s=14, marker='o', zorder=3,
                       facecolor=SCOLOR[col] if filled else 'white',
                       edgecolor=SCOLOR[col], linewidth=0.9)
    ax.axvline(0, color='#888888', lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f'Optimization arm vs {refname}', loc='left', fontsize=8,
                 fontweight='bold')
    ax.grid(axis='x')
    ax.grid(axis='y', linewidth=0.3)
    ax.set_xlabel('rank-biserial correlation of the paired differences\n'
                  '(positive = higher for the optimization arm)')
    shandles = ([Line2D([], [], ls='', marker='o', mfc=SCOLOR[c], mec=SCOLOR[c],
                        ms=5, label=lab) for c, lab in METRICS] +
                [Line2D([], [], ls='', marker='o', mfc='#555', mec='#555', ms=5,
                        label='Holm p < 0.05 (filled)'),
                 Line2D([], [], ls='', marker='o', mfc='white', mec='#555', ms=5,
                        label='not significant (open)')])
    fig.legend(handles=shandles, loc='lower center', ncol=3, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.01), columnspacing=0.9,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, f'optimizations_stats_{tag}')
report.close()
print('all charts generated')
