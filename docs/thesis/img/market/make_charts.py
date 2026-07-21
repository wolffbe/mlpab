"""Generate all market assessment charts for the MLPAB thesis.
Fonts: prefers Times New Roman; falls back to Liberation Serif (metric-compatible)."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'MLPlatformAgentBench market assessment.csv'
OUT = HERE

df = pd.read_csv(SRC)
# the csv export repeats the stage headers per interface; pandas mangles the
# duplicates to feature.1/.2/.3 — map them back to the sheet's "feature 2/3/4"
df = df.rename(columns={f'{c}.{i}': f'{c} {i + 1}'
                        for c in ['feature', 'training', 'inference', 'ops']
                        for i in (1, 2, 3)})

def pts(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s == 'yes': return 1.0
        if s == 'partial': return 0.5
    return 0.0

def pts_url(v):
    """Score cells that hold either yes/partial/- or a documentation URL."""
    if isinstance(v, str) and v.strip().lower().startswith('http'):
        return 1.0
    return pts(v)

groups = {
    'Python SDK': ['feature', 'training', 'inference', 'ops'],
    'CLI': ['feature 2', 'training 2', 'inference 2', 'ops 2'],
    'MCP': ['feature 3', 'training 3', 'inference 3', 'ops 3',
            'model context protocol server hosting'],
    'ML platform agents': ['feature 4', 'training 4', 'inference 4', 'ops 4',
                   'PySpark code', 'file system', 'dashboards'],
}
MAX = 20.0
for g, cols in groups.items():
    df[g] = df[cols].map(pts_url).sum(axis=1)
df['total'] = df[list(groups)].sum(axis=1) / MAX
df['name'] = df['company'].replace({
    'Google Gemini Enterprise Agent Platform (formerly Vertex AI)': 'Google Gemini Enterprise Agent Platform'})
# acquired platforms carry their acquirer in brackets
ACQUIRERS = {'Dremio': 'SAP', 'Altair': 'Siemens',
             'Iguazio': 'McKinsey & Company'}
for comp, acq in ACQUIRERS.items():
    df.loc[df['company'] == comp, 'name'] += f' ({acq})'
df['hq'] = df['HQ'].replace({'Sweden': 'SE', 'Switzerland': 'CH'})

# single per-platform facts shown as a matrix between the names and the bars:
# yes/partial/no criteria as dots, location as text
MATRIX = [('active', 'Active', 'dot'),
          ('hq', 'Headquarters', 'text'),
          ('public SaaS', 'Public SaaS', 'dot'),
          ('agentic AI', 'ML Platform Agent', 'dot'),
          ('own-agent infrastructure', 'Agent deployments', 'dot'),
          ('meta harness', 'Meta-Harness', 'dot'),
          ('model context protocol server hosting', 'MCP server hosting', 'dot'),
          ('skills', 'Skills', 'dot')]
# dot count breaks ties between equal bar totals
df['dots'] = sum((df[col].map(pts_url) > 0).astype(int)
                 for col, _, kind in MATRIX if kind == 'dot')

colors = {'Python SDK': '#4477AA', 'CLI': '#66CCEE',
          'MCP': '#EE6677', 'ML platform agents': '#CCBB44'}

plt.rcParams.update({
    'pdf.fonttype': 42,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'FreeSerif'],
    'axes.spines.top': False, 'axes.spines.right': False,
})

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def pct(v):
    """Percentage label with round-half-up, so 92.5 prints as 93."""
    return f'{int(np.floor(v + 0.5))}%'

def pct_scale(vmax=None):
    """Axis limit and ticks: full scale, or the next 10% step above vmax."""
    lim = 100 if vmax is None else min(100, int(np.ceil(vmax / 10)) * 10)
    step = 10 if lim <= 60 else 20
    return lim, range(0, lim + 1, step)

def pct_axis(ax, xlabel, vmax=None):
    lim, ticks = pct_scale(vmax)
    ax.set_xlim(0, lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t}%' for t in ticks])
    ax.set_xlabel(xlabel)

DOT = {True: dict(facecolor='#555555', edgecolor='#555555'),
       False: dict(facecolor='white', edgecolor='#999999')}

def coverage_chart(d, fname, height, legend_y):
    d = d.sort_values(['total', 'dots'], ascending=True)
    fig, (axm, ax) = plt.subplots(1, 2, figsize=(6.2, height),
                                  width_ratios=[1.35, 2.9])
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for g in groups:
        vals = d[g].values.astype(float) / MAX * 100
        ax.barh(y, vals, left=left, height=0.62, color=colors[g],
                label=f'{g} ({len(groups[g])}/{MAX:.0f})',
                edgecolor='white', linewidth=0.4)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis='x', color='#E6E6E6', linewidth=0.6)
    pct_axis(ax, 'Market coverage', vmax=left.max())
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y),
              ncol=2, frameon=False, fontsize=8, columnspacing=1.2, handlelength=1.4)
    for yi, tot in enumerate(left):
        ax.text(tot + 0.8, yi, pct(tot), va='center', fontsize=7.5, color='#444444')
    for k, (col, _, kind) in enumerate(MATRIX):
        if kind == 'dot':
            for yi, v in enumerate(d[col].map(pts_url)):
                axm.scatter(k, yi, s=22, linewidths=0.8, zorder=3, **DOT[v > 0])
        else:
            for yi, t in enumerate(d[col]):
                axm.text(k, yi, t, va='center', ha='center',
                         fontsize=6.5, color='#444444')
    axm.set_xticks(range(len(MATRIX)))
    axm.set_xticklabels([lab for _, lab, _ in MATRIX],
                        rotation=60, ha='left', fontsize=7)
    axm.xaxis.set_ticks_position('top')
    axm.tick_params(axis='y', pad=7)
    axm.set_xlim(-0.55, len(MATRIX) - 0.55)
    axm.set_yticks(y); axm.set_yticklabels(d['name']); axm.grid(False)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.tick_params(length=0)
    for a in (ax, axm):
        a.set_ylim(-0.55, len(d) - 0.45)
    dots = [plt.Line2D([], [], ls='', marker='o', ms=5, mew=0.8,
                       mfc=DOT[v]['facecolor'], mec=DOT[v]['edgecolor'], label=l)
            for v, l in [(True, 'Yes'), (False, 'No')]]
    axm.legend(handles=dots, loc='upper center', bbox_to_anchor=(0.15, legend_y),
               ncol=1, frameon=False, fontsize=7, columnspacing=0.9,
               handletextpad=0.4)
    fig.tight_layout(w_pad=0.3); save(fig, fname)

def coverage_chart_t(d, fname):
    """Transposed variant: platforms as columns, fact matrix below the bars."""
    d = d.sort_values(['total', 'dots'], ascending=False)
    n = len(d)
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(6.2, 4.6),
                                  height_ratios=[2.6, 1.0])
    x = np.arange(n)
    bottom = np.zeros(n)
    for g in groups:
        vals = d[g].values.astype(float) / MAX * 100
        ax.bar(x, vals, bottom=bottom, width=0.62, color=colors[g],
               label=f'{g} ({len(groups[g])}/{MAX:.0f})',
               edgecolor='white', linewidth=0.4)
        bottom += vals
    for xi, tot in enumerate(bottom):
        ax.text(xi, tot + 2, pct(tot), ha='center', va='bottom',
                fontsize=7.5, color='#444444')
    lim, ticks = pct_scale(bottom.max())
    ax.set_ylim(0, lim)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t}%' for t in ticks])
    ax.set_ylabel('Market coverage')
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color='#E6E6E6', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    # legend above the axes so it cannot collide with the bar total labels
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
              borderaxespad=0.1, frameon=False, fontsize=7.5,
              columnspacing=1.4, handlelength=1.4)
    # fact matrix below, criteria as rows sharing the platform columns
    ym = np.arange(len(MATRIX))[::-1]
    for r, (col, _, kind) in zip(ym, MATRIX):
        if kind == 'dot':
            for xi, v in enumerate(d[col].map(pts_url)):
                axm.scatter(xi, r, s=22, linewidths=0.8, zorder=3, **DOT[v > 0])
        else:
            for xi, t in enumerate(d[col]):
                axm.text(xi, r, t, va='center', ha='center',
                         fontsize=6.5, color='#444444')
    axm.set_yticks(ym); axm.set_yticklabels([lab for _, lab, _ in MATRIX],
                                            fontsize=7)
    axm.set_ylim(-0.6, len(MATRIX) - 0.4)
    axm.set_xticks(x)
    axm.set_xticklabels(d['name'].str.replace(
        'Google Gemini Enterprise Agent Platform', 'Google Gemini\nEnterprise Agent Platform'),
        rotation=20, ha='right', fontsize=7)
    axm.grid(False)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.tick_params(length=0)
    for a in (ax, axm):
        a.set_xlim(-0.6, n - 0.4)
    dots = [plt.Line2D([], [], ls='', marker='o', ms=5, mew=0.8,
                       mfc=DOT[v]['facecolor'], mec=DOT[v]['edgecolor'], label=l)
            for v, l in [(True, 'Yes'), (False, 'No')]]
    axm.legend(handles=dots, loc='center left', bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=7, handletextpad=0.4)
    fig.tight_layout(h_pad=0.5); save(fig, fname)

top5 = df.nlargest(5, 'total')
coverage_chart_t(top5, 'coverage_top5')
coverage_chart(df.drop(top5.index), 'coverage_rest', 8.2, -0.075)

# Key finding: coverage by interface class, split by lifecycle stage; MCP
# hosting and skills are single criteria, shown as single-bar rows
STAGES = ['Feature', 'Training', 'Inference', 'Ops']
stage_colors = {'Feature': '#4477AA', 'Training': '#EE6677',
                'Inference': '#CCBB44', 'Ops': '#66CCEE'}
ifrows = {'Python SDK': dict(zip(STAGES, groups['Python SDK'])),
          'CLI': dict(zip(STAGES, groups['CLI'])),
          'MCP tools': dict(zip(STAGES, ['feature 3', 'training 3',
                                         'inference 3', 'ops 3'])),
          'MCP server hosting': {None: 'model context protocol server hosting'},
          'ML platform agents': dict(zip(STAGES, ['feature 4', 'training 4',
                                          'inference 4', 'ops 4'])),
          'Agent deployments': {None: 'own-agent infrastructure'},
          'Meta-Harness': {None: 'meta harness'},
          'Skills': {None: 'skills'}}
N = len(df)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
names = list(ifrows)
y0 = np.arange(len(names))[::-1]
h = 0.17
for r, (name, stages) in zip(y0, ifrows.items()):
    for k, (stage, col) in enumerate(stages.items()):
        v = df[col].map(pts_url).sum() / N * 100
        yy = r + (len(stages) - 1) / 2 * h - k * h
        b = ax.barh(yy, v, height=h * 0.92,
                    color=stage_colors[stage] if stage else '#BBBBBB')
        ax.bar_label(b, labels=[pct(v)], padding=3, fontsize=7, color='#444444')
ax.set_yticks(y0); ax.set_yticklabels(names)
ax.margins(y=0.02)
ax.set_axisbelow(True)
ax.grid(True, axis='x', color='#E6E6E6', linewidth=0.6)
vmax = max(df[col].map(pts_url).sum() / N * 100
           for stages in ifrows.values() for col in stages.values())
pct_axis(ax, 'Share of the 39 assessed platforms (full = 1, partial = 0.5)',
         vmax=vmax)
handles = ([plt.Rectangle((0, 0), 1, 1, color=stage_colors[s]) for s in STAGES]
           + [plt.Rectangle((0, 0), 1, 1, color='#BBBBBB')])
ax.legend(handles, STAGES + ['Single criterion'], loc='center right',
          frameon=False, fontsize=8)
fig.tight_layout(); save(fig, 'coverage_by_interface')
print('all charts generated')
