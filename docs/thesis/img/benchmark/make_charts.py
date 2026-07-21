"""Generate the RQ2 benchmark charts for the MLPAB thesis.

Ports the print-sized figures of results/results.ipynb section 8, scoped to
the committed RQ2 arms: Hopsworks and Databricks full grids (treatments 1-4
and 18-21). Fable, GCP, and the RQ3 optimization arms are out of scope here.
Sized for the KTH template text block (130 mm, included at width=linewidth).
The fonts prefer Times New Roman and fall back to Liberation Serif."""
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

# Cleaning as in the results notebook: keep the latest attempt per combination.
df = raw.sort_values('n').drop_duplicates(
    subset=['config', 'interface', 'skills', 'category', 'task'], keep='last')

# Keep the 22 platform-grounded benchmark tasks analyzed in the thesis.
EXCLUDED_TASKS = {'leakage', 'skew', 'drift', 'prediction_monitoring'}
df = df[~df.task.isin(EXCLUDED_TASKS)].copy()

# A killed agent does not emit final usage/timing metadata. Such rows remain
# failed accuracy observations, but their recorded zero turns and local time
# are placeholders rather than measurements and must not enter efficiency
# aggregates or paired tests.
df['timeout'] = df['error'].astype(str).str.contains('agent exited 124', na=False)
df.loc[df['timeout'], ['llm_calls', 'local_time_s']] = np.nan

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

# cost_usd in results.csv is cache inclusive for the Claude rows, priced by
# the framework from the retained session transcripts. Mistral cache usage was
# not retained, so Mistral cost stays a lower bound and is excluded from cost
# inference below
df['cost_ok'] = df.model.astype(str).str.startswith('claude-')

VARIANTS = [('cli', 'none'), ('cli', 'official'), ('sdk', 'none'), ('sdk', 'official')]

# skills were never delivered to the Mistral agent (bundle discovery-location
# bug), so Mistral appears only in the no-skills condition in every figure and
# contrast group. Its nominal official rows stay in results.csv as provenance
def variants_for(m):
    return [(i, sk) for i, sk in VARIANTS
            if not (m.startswith('mistral') and sk == 'official')]

def strata_for(m):
    return ['none'] if m.startswith('mistral') else ['none', 'official']
VLABEL = {('cli', 'none'): 'CLI no skills', ('cli', 'official'): 'CLI + skills',
          ('sdk', 'none'): 'SDK no skills', ('sdk', 'official'): 'SDK + skills'}
VCOLOR = {('cli', 'none'): '#90a4ae', ('cli', 'official'): '#455a64',
          ('sdk', 'none'): '#42a5f5', ('sdk', 'official'): '#1565c0'}
MCOLOR = {'opus': '#1565c0', 'sonnet': '#00897b',
          'mistral-large': '#ef6c00', 'mistral-medium': '#ad1457'}
MMARK = {'cli': 'o', 'sdk': '^'}
CATEGORIES = ['feature', 'training', 'inference', 'ops', 'capstone']
BENCHMARK_TASKS = df[['category', 'task']].drop_duplicates()
EXPECTED_TASKS = len(BENCHMARK_TASKS)
EXPECTED_TASKS_BY_CATEGORY = BENCHMARK_TASKS.groupby('category').size().to_dict()
INCOMPLETE_HATCH = '////'

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
    """Mean assertion pass fraction with invalid runs scored as zero.

    This is the pre-registered accuracy aggregation rule. Return NaN when there
    are no runs.
    """
    return d.pass_rate.where(d.valid, 0.0).mean() if len(d) else np.nan


def legend_handles():
    return ([Patch(facecolor=VCOLOR[v], label=VLABEL[v]) for v in VARIANTS] +
            [Patch(facecolor='white', edgecolor='#555', hatch=INCOMPLETE_HATCH,
                   label='incomplete, n/planned')])


def mark_and_annotate_bar(ax, bars, h, observed, expected, y_offset, fontsize):
    """Label a pass-rate bar and visibly mark incomplete observations."""
    if pd.isna(h):
        return
    b = bars[0]
    incomplete = observed < expected
    if incomplete:
        b.set_hatch(INCOMPLETE_HATCH)
        b.set_edgecolor('#444')
        b.set_linewidth(0.7)
        label = f'{h:.0%}\nn={observed}/{expected}'
        if h == 0:
            ax.plot(b.get_x() + b.get_width() / 2, 0.012, marker='x',
                    color='#444', markersize=3.5, markeredgewidth=0.8,
                    clip_on=False, zorder=4)
    else:
        label = f'{h:.0%}'
    ax.annotate(label, (b.get_x() + b.get_width() / 2, h + y_offset),
                ha='center', va='bottom', fontsize=fontsize, rotation=90,
                color='#333')


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


# fig: overview, pass fraction per model, one figure per platform
w = 0.19
for plat in PLATFORMS:
    fig, ax = plt.subplots(figsize=(TW, 2.9))
    x = np.arange(len(MODEL_ORDER))
    for xi, m in enumerate(MODEL_ORDER):
        vs = variants_for(m)
        for j_, (iface, sk) in enumerate(vs):
            observed = sel(CFG[plat][m], iface, sk)
            h = rate(observed)
            bars = ax.bar(xi + (j_ - (len(vs) - 1) / 2) * w, h, w * 0.92,
                          color=VCOLOR[(iface, sk)])
            mark_and_annotate_bar(ax, bars, h, len(observed), EXPECTED_TASKS,
                                  y_offset=0.01, fontsize=5.5)
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

# fig: category, one figure each for feature, training, inference, operations,
# and capstone tasks, with both platforms side by side
for cat in CATEGORIES:
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.3), sharey=True)
    for c, plat in enumerate(PLATFORMS):
        ax = axes[c]
        x = np.arange(len(MODEL_ORDER))
        for xi, m in enumerate(MODEL_ORDER):
            vs = variants_for(m)
            for j_, (iface, sk) in enumerate(vs):
                observed = sel(CFG[plat][m], iface, sk, category=cat)
                h = rate(observed)
                bars = ax.bar(xi + (j_ - (len(vs) - 1) / 2) * w, h, w * 0.92,
                              color=VCOLOR[(iface, sk)])
                mark_and_annotate_bar(
                    ax, bars, h, len(observed), EXPECTED_TASKS_BY_CATEGORY[cat],
                    y_offset=0.02, fontsize=4.2)
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

# fig: per-task heatmaps, one per platform x interface (appendix, one page each)
TASK_ORDER = []
for cat in CATEGORIES:
    TASK_ORDER += [(cat, t) for t in sorted(df[df.category == cat].task.unique())]
tcmap = mpl.colormaps['Blues'].copy()
tcmap.set_bad('#eeeeee')
for plat in PLATFORMS:
    for iface in ['cli', 'sdk']:
        cols = [(m, sk) for m in MODEL_ORDER for sk in strata_for(m)]
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
        for m in MODEL_ORDER:
            ks = [k for k, (m_, _) in enumerate(cols) if m_ == m]
            ax.text(sum(ks) / len(ks), -0.03, m.replace('mistral-', 'mistral-\n'),
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
        for k in range(1, len(cols)):
            if cols[k][0] != cols[k - 1][0]:
                ax.axvline(k - 0.5, color='white', lw=1.6)
        start = 0
        for cat in CATEGORIES:
            n_ = sum(1 for c_, _ in TASK_ORDER if c_ == cat)
            if start > 0:
                ax.axhline(start - 0.5, color='white', lw=1.6)
            ax.text(-0.2, (2 * start + n_ - 1) / 2, cat, rotation=90, ha='right',
                    va='center', fontsize=7, fontweight='bold',
                    transform=ax.get_yaxis_transform(), color='#444')
            start += n_
        ax.tick_params(length=0)
        # the skill-condition key (\u2013 / +sk) is explained in the figure captions
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        cb.ax.tick_params(labelsize=6)
        fig.tight_layout()
        save(fig, f'benchmark_tasks_{plat}_{iface}')

# fig: efficiency frontier, 2 metrics x 2 platforms
fr_rows = []
for plat in PLATFORMS:
    for m in MODEL_ORDER:
        for iface, sk in variants_for(m):
            s = sel(CFG[plat][m], iface, sk)
            if len(s) == 0:
                continue
            solved = int(s.success.sum())
            turn_total = s.llm_calls.sum(min_count=len(s))
            fr_rows.append({
                'platform': plat, 'model': m, 'interface': iface, 'skills': sk,
                'solve_rate': solved / len(s),
                'cost_per_solve': s.cost_usd.sum() / solved if solved else np.nan,
                'turns_per_solve': turn_total / solved if solved else np.nan,
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
                    label='+ skills, filled'),
             Line2D([], [], ls='', marker='o', mfc='white', mec='#555',
                    label='no skills, open')])
for xcol, xlabel, tag in [('cost_per_solve', 'model invocation cost per solved task in USD, logarithmic scale', 'cost'),
                          ('turns_per_solve', 'LLM turns per solved task, logarithmic scale', 'turns')]:
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

# fig: planned contrasts, Wilcoxon rank-biserial effect sizes, Holm-adjusted
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
METRICS = [('pr0', 'pass fraction'), ('cost_usd', 'model cost'), ('llm_calls', 'turns')]
PSHORT = {'hopsworks': 'hw', 'databricks': 'db'}

from itertools import combinations

def cellf(plat, m, iface, sk):
    return df[(df.config == CFG[plat][m]) & (df.interface == iface) &
              (df.skills == sk)].set_index(['category', 'task'])

PSH = {'hopsworks': 'hw', 'databricks': 'db'}
SKL = {'none': '-', 'official': '+sk'}
# interface and skills contrasts pair identical seeded instances with a shared
# run_id. Platform and model contrasts pair task templates across instances
contrast_groups = {}
contrast_groups['Interface, SDK vs CLI'] = [
    (f'{PSH[p]} {m} {SKL[sk]}', cellf(p, m, 'cli', sk), cellf(p, m, 'sdk', sk))
    for p in PLATFORMS for m in MODEL_ORDER for sk in strata_for(m)]
# skills bundles were installed under .claude/skills/, which the Mistral Vibe
# agent never reads (session logs list no platform skill, 0 skill calls in all
# nominal Mistral official rows). The official condition was a no-op for Mistral,
# so the skills contrasts are restricted to the Claude cells where the treatment
# was actually delivered
contrast_groups['Skills, official vs none'] = [
    (f'{PSH[p]} {m} {i}', cellf(p, m, i, 'none'), cellf(p, m, i, 'official'))
    for p in PLATFORMS for m in MODEL_ORDER for i in ['cli', 'sdk']
    if not m.startswith('mistral')]
contrast_groups['Platform, Databricks vs Hopsworks'] = [
    (f'{m} {i} {SKL[sk]}', cellf('hopsworks', m, i, sk), cellf('databricks', m, i, sk))
    for m in MODEL_ORDER for i in ['cli', 'sdk'] for sk in strata_for(m)]

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

def cost_eligible(a, b):
    return bool(a['cost_ok'].all() and b['cost_ok'].all())


def mcnemar_success(a, b):
    common = a.index.intersection(b.index)
    x = a.loc[common, 'success'].astype(bool)
    y = b.loc[common, 'success'].astype(bool)
    n01 = int((~x & y).sum())
    n10 = int((x & ~y).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10, len(common)
    p = sps.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return float(min(1.0, p)), n01, n10, len(common)


def paired_counts(a, b, col):
    common = a.index.intersection(b.index)
    x = a.loc[common, col].astype(float)
    y = b.loc[common, col].astype(float)
    ok = x.notna() & y.notna()
    d = y[ok] - x[ok]
    return int(ok.sum()), int((d != 0).sum())


def group_results(contrasts):
    """Adjust every inferential test in one manipulated-factor contrast group."""
    tested = {}
    keys, ps = [], []
    for col, _ in METRICS:
        rows = []
        for i, (_, a, b) in enumerate(contrasts):
            result = (None if col == 'cost_usd' and not cost_eligible(a, b)
                      else wtest2(a, b, col))
            rows.append(result)
            if result is not None:
                keys.append((col, i))
                ps.append(result[0])
        tested[col] = rows
    mcnemar = [mcnemar_success(a, b) for _, a, b in contrasts]
    for i, result in enumerate(mcnemar):
        keys.append(('success', i))
        ps.append(result[0])
    adjusted = dict(zip(keys, holm(ps)))
    result = {
        col: [(entry[1], adjusted[(col, i)]) if entry is not None else None
              for i, entry in enumerate(rows)]
        for col, rows in tested.items()
    }
    result['success'] = [(*entry[1:], adjusted[('success', i)])
                         for i, entry in enumerate(mcnemar)]
    result['tests'] = len(ps)
    return result


def draw_group(ax, contrasts):
    res = group_results(contrasts)
    y = np.arange(len(contrasts))[::-1]
    for col, _ in METRICS:
        for yi, entry in zip(y, res[col]):
            if entry is None:
                continue
            rb, padj = entry
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

METRICS = [('pr0', 'pass fraction'), ('cost_usd', 'model cost'), ('llm_calls', 'turns'),
           ('local_time_s', 'time')]
SCOLOR = {'pr0': '#1565c0', 'cost_usd': '#ef6c00', 'llm_calls': '#00897b',
          'local_time_s': '#8e24aa'}
SOFF = {'pr0': 0.3, 'cost_usd': 0.1, 'llm_calls': -0.1, 'local_time_s': -0.3}

FTAG = {'Interface, SDK vs CLI': 'interface',
        'Skills, official vs none': 'skills',
        'Platform, Databricks vs Hopsworks': 'platform'}
for group, contrasts in contrast_groups.items():
    fig, ax = plt.subplots(figsize=(TW, 0.165 * len(contrasts) + 1.55))
    draw_group(ax, contrasts)
    ax.set_title(group, loc='left', fontsize=8, fontweight='bold')
    # the reading direction and the +sk key are explained in the figure captions
    ax.set_xlabel('rank-biserial correlation of the paired differences')
    shandles = ([Line2D([], [], ls='', marker='o', mfc=SCOLOR[c], mec=SCOLOR[c],
                        ms=5, label=lab) for c, lab in METRICS] +
                [Line2D([], [], ls='', marker='o', mfc='#555', mec='#555', ms=5,
                        label='Holm p < 0.05, filled'),
                 Line2D([], [], ls='', marker='o', mfc='white', mec='#555', ms=5,
                        label='not significant, open')])
    fig.legend(handles=shandles, loc='lower center', ncol=3, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.01), columnspacing=0.9,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, f'benchmark_stats_{FTAG[group]}')

# fig: pairwise model contrasts within platform x interface x skills
# Mistral has no official-condition cells (skills never delivered), so pairs at
# the official condition exist only between the Claude models
model_contrasts = [
    (f'{PSH[p]} {i} {SKL[sk]} {m2} vs {m1}', cellf(p, m1, i, sk), cellf(p, m2, i, sk))
    for p in PLATFORMS for i in ['cli', 'sdk'] for sk in ['none', 'official']
    for m1, m2 in combinations(MODEL_ORDER, 2)
    if not (sk == 'official'
            and (m1.startswith('mistral') or m2.startswith('mistral')))]
fig, ax = plt.subplots(figsize=(TW, 0.165 * len(model_contrasts) + 1.7))
draw_group(ax, model_contrasts)
ax.set_title('Model, second named vs first named', loc='left', fontsize=8,
             fontweight='bold')
ax.set_xlabel('rank-biserial correlation of the paired differences')
fig.legend(handles=shandles, loc='lower center', ncol=3, frameon=False,
           fontsize=6, bbox_to_anchor=(0.5, -0.005), columnspacing=0.9,
           handletextpad=0.4)
fig.tight_layout(rect=(0, 0.035, 1, 1))
save(fig, 'benchmark_stats_models')


# authoritative stats report: Wilcoxon (normal approximation), McNemar on success,
# paired t with Cohen's d as sensitivity. All reported numbers derive from here
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


all_groups = dict(contrast_groups)
all_groups['Model, pairwise'] = model_contrasts
with open(OUT / 'stats_report.txt', 'w') as fh:
    for group, contrasts in all_groups.items():
        fres = group_results(contrasts)
        fh.write(f'==== {group}, {len(contrasts)} contrasts, '
                 f'{fres["tests"]} tests under one factor-wide Holm adjustment\n')
        for col, lab in METRICS:
            elig = [not (col == 'cost_usd' and not cost_eligible(a, b))
                    for _, a, b in contrasts]
            res = [wtest2(a, b, col) if e else None
                   for (_, a, b), e in zip(contrasts, elig)]
            idx = [i for i, r_ in enumerate(res) if r_ is not None]
            padjs = {i: fres[col][i][1] for i in idx}
            tt = [ttest_d(a, b, col) if e else None
                  for (_, a, b), e in zip(contrasts, elig)]
            agree = sum(1 for r_, t_ in zip(res, tt)
                        if r_ and t_ and ((r_[1] >= 0) == (t_[1] >= 0) or r_[1] == 0))
            nsig = sum(1 for i in idx if padjs[i] < 0.05)
            fh.write(f'-- {lab}: wilcoxon {nsig}/{len(idx)} sig after factor-wide Holm '
                 f'{len(contrasts) - len(idx)} excluded because Mistral cost is unreliable, '
                     f't-test direction agreement {agree}/{len(idx)}\n')
            for i, (name, a, b) in enumerate(contrasts):
                if res[i] is None:
                    fh.write(f'   {name:34s} excluded because Mistral cost is not reliable\n')
                    continue
                n, nz = paired_counts(a, b, col)
                mark = '*' if padjs[i] < 0.05 else ' '
                fh.write(f'   {name:34s} rb={res[i][1]:+.2f} holm={padjs[i]:.4f}{mark} '
                         f'n={n} nz={nz} t_p={tt[i][0]:.4f} d={tt[i][1]:+.2f}\n')
        sres = fres['success']
        nsig = sum(1 for _, _, _, a_ in sres if a_ < 0.05)
        fh.write(f'-- mcnemar on success: {nsig}/{len(contrasts)} sig after factor-wide Holm\n')
        for (name, _, _), (n01, n10, n, a_) in zip(contrasts, sres):
            mark = ' *' if a_ < 0.05 else ''
            fh.write(f'   {name:34s} n={n} n01={n01} n10={n10} holm={a_:.4f}{mark}\n')
print('stats report written')
print('stats chart generated')
