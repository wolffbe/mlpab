"""Generate the RQ2 benchmark charts for the MLPAB thesis.

Ports the print-sized figures of results/results.ipynb section 8, scoped to
the committed RQ2 arms: Hopsworks and Databricks full grids (treatments 1-4
and 18-21). Fable, GCP, and the RQ3 optimization arms are out of scope here.
Sized for the KTH template text block (130 mm, included at width=linewidth).
Fonts: prefers Times New Roman; falls back to Liberation Serif."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / 'results' / 'results.csv'
OUT = HERE

raw = pd.read_csv(SRC)
for c in ['asserts_passed', 'total_asserts', 'local_time_s', 'cost_usd', 'llm_calls', 'n']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
raw['valid'] = raw['valid'].astype(str).str.lower() == 'true'
raw['success'] = raw['success'].astype(str).str.lower() == 'true'
raw['pass_rate'] = raw['asserts_passed'] / raw['total_asserts']

# cleaning as in the results notebook: keep the latest attempt per combo and
# drop grader-infra crashes (the agent was never judged)
df = raw.sort_values('n').drop_duplicates(
    subset=['config', 'interface', 'skills', 'category', 'task'], keep='last')
df = df[~df['error'].astype(str).str.contains('grader failed to run', na=False)].copy()

# RQ2 scope: the two committed platforms, full grids only
CFG = {
    'hopsworks': {
        'opus': '1_hw-full-cli-sdk-skills-opus',
        'mistral-large': '2_hw-full-cli-sdk-skills-mistral-large',
        'sonnet': '3_hw-full-cli-sdk-skills-sonnet',
        'mistral-medium': '4_hw-full-cli-sdk-skills-mistral-medium',
    },
    'databricks': {
        'opus': '18_db-full-cli-sdk-skills-opus',
        'mistral-large': '20_db-full-cli-sdk-skills-mistral-large',
        'sonnet': '19_db-full-cli-sdk-skills-sonnet',
        'mistral-medium': '21_db-full-cli-sdk-skills-mistral-medium',
    },
}
PLATFORMS = ['hopsworks', 'databricks']
MODEL_ORDER = ['opus', 'mistral-large', 'sonnet', 'mistral-medium']
df = df[df.config.isin([c for m in CFG.values() for c in m.values()])].copy()

VARIANTS = [('cli', 'none'), ('cli', 'official'), ('sdk', 'none'), ('sdk', 'official')]
VLABEL = {('cli', 'none'): 'CLI no skills', ('cli', 'official'): 'CLI + skills',
          ('sdk', 'none'): 'SDK no skills', ('sdk', 'official'): 'SDK + skills'}
VCOLOR = {('cli', 'none'): '#90a4ae', ('cli', 'official'): '#455a64',
          ('sdk', 'none'): '#42a5f5', ('sdk', 'official'): '#1565c0'}
MCOLOR = {'opus': '#1565c0', 'sonnet': '#00897b',
          'mistral-large': '#ef6c00', 'mistral-medium': '#ad1457'}
MMARK = {'cli': 'o', 'sdk': '^'}
CATEGORIES = ['feature', 'training', 'inference', 'ops', 'capstone']

TW = 5.12  # KTH text block width in inches (130 mm)
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


def sel(cfg, iface=None, sk=None, category=None, task=None):
    m = df.config == cfg
    if iface is not None:
        m &= df.interface == iface
    if sk is not None:
        m &= df.skills == sk
    if category is not None:
        m &= df.category == category
    if task is not None:
        m &= df.task == task
    return df[m]


def rate(d):
    """Mean assertion pass fraction, invalid runs scored as zero
    (the pre-registered accuracy aggregation rule); NaN when no runs."""
    return d.pass_rate.where(d.valid, 0.0).mean() if len(d) else np.nan


def legend_handles():
    return [Patch(facecolor=VCOLOR[v], label=VLABEL[v]) for v in VARIANTS]


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


# fig: overview — pass fraction per model, one figure per platform
w = 0.19
for plat in PLATFORMS:
    fig, ax = plt.subplots(figsize=(TW, 2.9))
    x = np.arange(len(MODEL_ORDER))
    for j_, (iface, sk) in enumerate(VARIANTS):
        vals = [rate(sel(CFG[plat][m], iface, sk)) for m in MODEL_ORDER]
        bars = ax.bar(x + (j_ - 1.5) * w, vals, w * 0.92, color=VCOLOR[(iface, sk)])
        for b, h in zip(bars, vals):
            if pd.notna(h):
                ax.annotate(f'{h:.0%}', (b.get_x() + b.get_width() / 2, h + 0.01),
                            ha='center', va='bottom', fontsize=5.5, rotation=90,
                            color='#333')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_ylabel('assertion pass fraction')
    ax.legend(handles=legend_handles(), ncol=4, frameon=False, fontsize=6,
              loc='lower center', bbox_to_anchor=(0.5, -0.32),
              columnspacing=1.2, handlelength=1.2)
    fig.tight_layout()
    save(fig, f'benchmark_overview_{plat}')

# fig: category — one figure per task family, both platforms side by side
for cat in CATEGORIES:
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.3), sharey=True)
    for c, plat in enumerate(PLATFORMS):
        ax = axes[c]
        x = np.arange(len(MODEL_ORDER))
        for j_, (iface, sk) in enumerate(VARIANTS):
            vals = [rate(sel(CFG[plat][m], iface, sk, category=cat))
                    for m in MODEL_ORDER]
            bars = ax.bar(x + (j_ - 1.5) * w, vals, w * 0.92,
                          color=VCOLOR[(iface, sk)])
            for b, h in zip(bars, vals):
                if pd.notna(h):
                    ax.annotate(f'{h:.0%}',
                                (b.get_x() + b.get_width() / 2, h + 0.02),
                                ha='center', va='bottom', fontsize=4.2,
                                rotation=90, color='#333')
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=35, ha='right', fontsize=6.5)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, .5, 1.0])
        ax.tick_params(labelsize=6.5)
        ax.set_title(plat.capitalize(), fontweight='bold')
        if c == 0:
            ax.set_ylabel('assertion pass fraction', fontsize=7)
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4,
               frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.2, handlelength=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, f'benchmark_category_{cat}')

# fig: per-task heatmaps — one per platform x interface (appendix, one page each)
TASK_ORDER = []
for cat in CATEGORIES:
    TASK_ORDER += [(cat, t) for t in sorted(df[df.category == cat].task.unique())]
tcmap = mpl.colormaps['Blues'].copy()
tcmap.set_bad('#eeeeee')
for plat in PLATFORMS:
    for iface in ['cli', 'sdk']:
        cols = [(m, sk) for m in MODEL_ORDER for sk in ['none', 'official']]
        M = np.full((len(TASK_ORDER), len(cols)), np.nan)
        for i, (cat, t) in enumerate(TASK_ORDER):
            for k, (m, sk) in enumerate(cols):
                M[i, k] = rate(sel(CFG[plat][m], iface, sk, task=t))
        fig, ax = plt.subplots(figsize=(TW, 7.6))
        ax.grid(False)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=tcmap, vmin=0, vmax=1,
                       aspect='auto')
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(['+sk' if sk == 'official' else '–' for _, sk in cols],
                           fontsize=6)
        for g, m in enumerate(MODEL_ORDER):
            ax.text(2 * g + 0.5, -0.03, m.replace('mistral-', 'mistral-\n'),
                    ha='center', va='top', fontsize=6.5,
                    transform=ax.get_xaxis_transform(), color='#333')
        ax.set_yticks(np.arange(len(TASK_ORDER)))
        ax.set_yticklabels([t for _, t in TASK_ORDER], fontsize=6.5)
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if not np.isnan(M[i, k]):
                    ax.text(k, i, f'{M[i, k] * 100:.0f}', ha='center', va='center',
                            fontsize=5.5,
                            color='white' if M[i, k] > 0.55 else '#222')
        for xg in range(2, len(cols), 2):
            ax.axvline(xg - 0.5, color='white', lw=1.6)
        start = 0
        for cat in CATEGORIES:
            n_ = sum(1 for c_, _ in TASK_ORDER if c_ == cat)
            if start > 0:
                ax.axhline(start - 0.5, color='white', lw=1.6)
            ax.text(-0.5, (2 * start + n_ - 1) / 2, cat, rotation=90, ha='right',
                    va='center', fontsize=7, fontweight='bold',
                    transform=ax.get_yaxis_transform(), color='#444')
            start += n_
        ax.tick_params(length=0)
        ax.set_xlabel('\u2013 = no skills    +sk = official skills', fontsize=6.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        cb.ax.tick_params(labelsize=6)
        fig.tight_layout()
        save(fig, f'benchmark_tasks_{plat}_{iface}')

# fig: efficiency frontier — 2 metrics x 2 platforms
fr_rows = []
for plat in PLATFORMS:
    for m in MODEL_ORDER:
        for iface, sk in VARIANTS:
            s = sel(CFG[plat][m], iface, sk)
            if len(s) == 0:
                continue
            solved = int(s.success.sum())
            fr_rows.append({
                'platform': plat, 'model': m, 'interface': iface, 'skills': sk,
                'solve_rate': solved / len(s),
                'cost_per_solve': s.cost_usd.sum() / solved if solved else np.nan,
                'turns_per_solve': s.llm_calls.sum() / solved if solved else np.nan,
            })
frontier = pd.DataFrame(fr_rows)


def pareto(sub, xcol):
    """Non-dominated cells: no other cell has higher solve_rate AND lower x."""
    keep = []
    for i, r in sub.iterrows():
        if pd.isna(r[xcol]):
            continue
        dominated = ((sub.solve_rate >= r.solve_rate) & (sub[xcol] < r[xcol]) &
                     ((sub.solve_rate > r.solve_rate) | (sub[xcol] < r[xcol]))).any()
        if not dominated:
            keep.append(i)
    return sub.loc[keep].sort_values(xcol)


fhandles = ([Line2D([], [], ls='', marker='s', color=MCOLOR[m], label=m)
             for m in MODEL_ORDER] +
            [Line2D([], [], ls='', marker=MMARK[i_], mfc='white', mec='#555',
                    label=i_.upper()) for i_ in ['cli', 'sdk']] +
            [Line2D([], [], ls='', marker='o', mfc='#555', mec='#555',
                    label='+ skills (filled)'),
             Line2D([], [], ls='', marker='o', mfc='white', mec='#555',
                    label='no skills (open)')])
for xcol, xlabel, tag in [('cost_per_solve', 'cost per solved task (USD, log)', 'cost'),
                          ('turns_per_solve', 'LLM turns per solved task (log)', 'turns')]:
    fig, axes = plt.subplots(1, 2, figsize=(TW, 3.2))
    for c_, plat in enumerate(PLATFORMS):
        ax = axes[c_]
        sub = frontier[frontier.platform == plat]
        for _, r in sub.iterrows():
            if pd.isna(r[xcol]):
                continue
            filled = r.skills == 'official'
            ax.scatter(r[xcol], r.solve_rate, s=26, marker=MMARK[r.interface],
                       facecolor=MCOLOR[r.model] if filled else 'white',
                       edgecolor=MCOLOR[r.model], linewidth=1.1, zorder=3)
        par = pareto(sub, xcol)
        ax.step(par[xcol], par.solve_rate, where='post', color='#9e9e9e',
                lw=0.9, zorder=2)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylim(0, 1.06)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(labelsize=6.5)
        ax.set_title(plat.capitalize(), fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=7)
        if c_ == 0:
            ax.set_ylabel('solve rate')
    fig.legend(handles=fhandles, loc='lower center', ncol=4, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.02), columnspacing=0.9,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, f'benchmark_frontier_{tag}')
print('all charts generated')

# fig: planned contrasts — Wilcoxon rank-biserial effect sizes, Holm-adjusted
from scipy import stats as sps


def holm(ps):
    p = np.array(ps, float)
    adj = np.full_like(p, np.nan)
    order = np.argsort(p)
    mx = 0.0
    for rank, i in enumerate(order):
        mx = max(mx, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = mx
    return adj


df['pr0'] = df.pass_rate.where(df.valid, 0.0)
METRICS = [('pr0', 'pass fraction'), ('cost_usd', 'cost'), ('llm_calls', 'turns')]
PSHORT = {'hopsworks': 'hw', 'databricks': 'db'}

from itertools import combinations

def cellf(plat, m, iface, sk):
    return df[(df.config == CFG[plat][m]) & (df.interface == iface) &
              (df.skills == sk)].set_index(['category', 'task'])

PSH = {'hopsworks': 'hw', 'databricks': 'db'}
SKL = {'none': '-', 'official': '+sk'}
# interface and skills contrasts pair identical seeded instances (shared
# run_id); platform and model contrasts pair task templates across instances
families = {}
families['Interface (SDK vs CLI)'] = [
    (f'{PSH[p]} {m} {SKL[sk]}', cellf(p, m, 'cli', sk), cellf(p, m, 'sdk', sk))
    for p in PLATFORMS for m in MODEL_ORDER for sk in ['none', 'official']]
families['Skills (official vs none)'] = [
    (f'{PSH[p]} {m} {i}', cellf(p, m, i, 'none'), cellf(p, m, i, 'official'))
    for p in PLATFORMS for m in MODEL_ORDER for i in ['cli', 'sdk']]
families['Platform (Databricks vs Hopsworks)'] = [
    (f'{m} {i} {SKL[sk]}', cellf('hopsworks', m, i, sk), cellf('databricks', m, i, sk))
    for m in MODEL_ORDER for i in ['cli', 'sdk'] for sk in ['none', 'official']]

def wtest2(a, b, col):
    common = a.index.intersection(b.index)
    x, y = a.loc[common, col].astype(float), b.loc[common, col].astype(float)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    d = y - x
    if len(d) == 0 or (d != 0).sum() == 0:
        return 1.0, 0.0
    p = sps.wilcoxon(x, y, zero_method='wilcox', method='approx').pvalue
    nz = d[d != 0].values
    r = sps.rankdata(abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(p), float(rb)

def draw_family(ax, contrasts):
    res = {}
    for col, _ in METRICS:
        tested = [wtest2(a, b, col) for _, a, b in contrasts]
        res[col] = list(zip([t[1] for t in tested], holm([t[0] for t in tested])))
    y = np.arange(len(contrasts))[::-1]
    for col, _ in METRICS:
        for yi, (rb, padj) in zip(y, res[col]):
            filled = padj < 0.05
            ax.scatter(rb, yi + SOFF[col], s=16, marker='o', zorder=3,
                       facecolor=SCOLOR[col] if filled else 'white',
                       edgecolor=SCOLOR[col], linewidth=0.9)
    ax.axvline(0, color='#888888', lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in contrasts], fontsize=6)
    ax.set_ylim(-0.7, len(contrasts) - 0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(axis='x')
    ax.grid(axis='y', linewidth=0.3)

METRICS = [('pr0', 'pass fraction'), ('cost_usd', 'cost'), ('llm_calls', 'turns'),
           ('local_time_s', 'time')]
SCOLOR = {'pr0': '#1565c0', 'cost_usd': '#ef6c00', 'llm_calls': '#00897b',
          'local_time_s': '#8e24aa'}
SOFF = {'pr0': 0.3, 'cost_usd': 0.1, 'llm_calls': -0.1, 'local_time_s': -0.3}

FTAG = {'Interface (SDK vs CLI)': 'interface',
        'Skills (official vs none)': 'skills',
        'Platform (Databricks vs Hopsworks)': 'platform'}
for fam, contrasts in families.items():
    fig, ax = plt.subplots(figsize=(TW, 0.165 * len(contrasts) + 1.55))
    draw_family(ax, contrasts)
    ax.set_title(fam, loc='left', fontsize=8, fontweight='bold')
    ax.set_xlabel('rank-biserial correlation of the paired differences\n'
                  '(positive = higher for the condition named first in the title, '
                  '+sk = official skills)')
    shandles = ([Line2D([], [], ls='', marker='o', mfc=SCOLOR[c], mec=SCOLOR[c],
                        ms=5, label=lab) for c, lab in METRICS] +
                [Line2D([], [], ls='', marker='o', mfc='#555', mec='#555', ms=5,
                        label='Holm p < 0.05 (filled)'),
                 Line2D([], [], ls='', marker='o', mfc='white', mec='#555', ms=5,
                        label='not significant (open)')])
    fig.legend(handles=shandles, loc='lower center', ncol=3, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.01), columnspacing=0.9,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, f'benchmark_stats_{FTAG[fam]}')

# fig: model family — pairwise model contrasts within platform x interface x skills
model_contrasts = [
    (f'{PSH[p]} {i} {SKL[sk]} {m2} vs {m1}', cellf(p, m1, i, sk), cellf(p, m2, i, sk))
    for p in PLATFORMS for i in ['cli', 'sdk'] for sk in ['none', 'official']
    for m1, m2 in combinations(MODEL_ORDER, 2)]
fig, ax = plt.subplots(figsize=(TW, 9.2))
draw_family(ax, model_contrasts)
ax.set_title('Model (second named vs first named)', loc='left', fontsize=8,
             fontweight='bold')
ax.set_xlabel('rank-biserial correlation of the paired differences\n'
              '(positive = higher for the model named second, +sk = official skills)')
fig.legend(handles=shandles, loc='lower center', ncol=3, frameon=False,
           fontsize=6, bbox_to_anchor=(0.5, -0.005), columnspacing=0.9,
           handletextpad=0.4)
fig.tight_layout(rect=(0, 0.035, 1, 1))
save(fig, 'benchmark_stats_models')


# authoritative stats report: Wilcoxon (auto method), McNemar on success,
# paired t with Cohen's d as sensitivity — all reported numbers derive from here
def mcnemar_success(a, b):
    common = a.index.intersection(b.index)
    x = a.loc[common, 'success'].astype(bool)
    y = b.loc[common, 'success'].astype(bool)
    n01 = int((~x & y).sum())
    n10 = int((x & ~y).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(min(1.0, sps.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)), n01, n10


def ttest_d(a, b, col):
    common = a.index.intersection(b.index)
    x = a.loc[common, col].astype(float)
    y = b.loc[common, col].astype(float)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    d = y - x
    if len(d) < 2 or d.std() == 0:
        return 1.0, 0.0
    return float(sps.ttest_rel(y, x).pvalue), float(d.mean() / d.std())


allfam = dict(families)
allfam['Model (pairwise)'] = model_contrasts
with open(OUT / 'stats_report.txt', 'w') as fh:
    for fam, contrasts in allfam.items():
        fh.write(f'==== {fam} ({len(contrasts)} contrasts)\n')
        for col, lab in METRICS:
            res = [wtest2(a, b, col) for _, a, b in contrasts]
            adj = holm([r[0] for r in res])
            tt = [ttest_d(a, b, col) for _, a, b in contrasts]
            agree = sum(1 for (wp, rb), (tp, d) in zip(res, tt)
                        if (rb >= 0) == (d >= 0) or rb == 0)
            nsig = sum(1 for a_ in adj if a_ < 0.05)
            fh.write(f'-- {lab}: wilcoxon {nsig}/{len(contrasts)} sig after Holm; '
                     f't-test direction agreement {agree}/{len(contrasts)}\n')
            for (name, _, _), (wp, rb), a_, (tp, d) in zip(contrasts, res, adj, tt):
                mark = '*' if a_ < 0.05 else ' '
                fh.write(f'   {name:34s} rb={rb:+.2f} holm={a_:.4f}{mark} t_p={tp:.4f} d={d:+.2f}\n')
        mres = [mcnemar_success(a, b) for _, a, b in contrasts]
        madj = holm([r[0] for r in mres])
        nsig = sum(1 for a_ in madj if a_ < 0.05)
        fh.write(f'-- mcnemar on success: {nsig}/{len(contrasts)} sig after Holm\n')
        for (name, _, _), (mp, n01, n10), a_ in zip(contrasts, mres, madj):
            if a_ < 0.05:
                fh.write(f'   {name:34s} n01={n01} n10={n10} holm={a_:.4f}*\n')
print('stats report written')
print('stats chart generated')
