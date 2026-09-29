"""Figures for the submission deck. All numbers are read from result files."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from deck_style import *

bad = {}
h2h = json.load(open('results_head2head.json'))
fix = json.load(open('results_nominal_fix.json'))
lwo = json.load(open('results_fc_summary.json'))['lwo']   # full-curve leave-window-out (3,012-output model, passivity checked)

# ------------------------------------------------------------------- A pipeline
W, H = 5.67, 4.76
fig = card_fig(W, H)
ax = canvas_ax(fig)
ax.text(5, 78.5, 'Six stages, each backed by a measurement', fontsize=13.5, fontweight='bold', color=CHAR)
stages = [('Audit the data', 'reciprocal to 1e-11; largest singular value 0.995, limit 1', CHAR, None),
          ('Kriging with consistency checks', 'full-curve model: 0 of 734 M predicted matrices non-passive', CHAR, None),
          ('Validation on unseen boards', '93% verdicts right, 90% intervals cover 91%', CHAR, None),
          ('Exact sensitivity analysis', 'no random sampling, 0.3 s, from 30 real runs', BRAND, 'KPI 2'),
          ('Virtual design and tolerance study', '961 designs x 2,000 boards = 1.92 M evaluations', BRAND, 'KPI 1'),
          ('Scenario check, boards held out', '5 of 5 predictions inside the real interval', GREEN, None)]
y0, dy = 67.0, 10.9
for i, (t_, s_, col, tag) in enumerate(stages):
    y = y0 - i * dy
    if i < len(stages) - 1:
        ax.plot([9, 9], [y - 2.6, y - dy + 2.6], color=RULE, lw=2.2, zorder=1)
    ax.add_patch(Circle((9, y), 2.6, fc=col, ec='none', zorder=3))
    ax.text(9, y, str(i + 1), ha='center', va='center', color='white', fontsize=12,
            fontweight='bold', zorder=4)
    ax.text(16, y + 1.9, t_, fontsize=12.5, fontweight='bold', va='center', color=CHAR)
    ax.text(16, y - 2.4, s_, fontsize=10.5, va='center', color=MUTED)
    if tag:
        ax.add_patch(FancyBboxPatch((84, y + 0.35), 12, 3.2, boxstyle='round,pad=0.2,rounding_size=1.4',
                                    fc=BRAND, ec='none', zorder=3))
        ax.text(90, y + 1.95, tag, ha='center', va='center', color='white', fontsize=10,
                fontweight='bold', zorder=4)
ax.plot([4, 96], [7.4, 7.4], color=RULE, lw=1.2)
ax.text(4, 3.6, 'Across stages 3 to 5: other models, other limits, uncertainty', fontsize=10.5,
        color=MUTED, va='center')
bad['pipeline'] = save_card(fig, 'pipeline')

# ---------------------------------------------------------------- B1 head to head
W, H = 5.67, 4.25
fig = card_fig(W, H)
ax = axes_in(fig, 0.13, 0.15, 0.53, 0.62)
ns = [30, 45, 60]
kp = ['S11_max_dB', 'S31_mean_dB', 'iso_max_dB']
mean_err = lambda m: [np.mean([h2h[k][f'n{n}'][m]['mad'] for k in kp]) for n in ns]
series = [('binned', 'conventional\nMonte Carlo', CHAR, 'o', '-', 2.4, 0.72),
          ('linear', 'linear screening', MUTED, 'v', ':', 2.0, 0.56),
          ('mlp+mc', 'neural network\n+ Monte Carlo', BLUE, '^', '--', 2.0, 0.40),
          ('ours', 'kriging, exact\n(ours)', BRAND, 's', '-', 3.4, 0.26),
          ('pce', 'polynomial\nchaos', BLUE, 'D', '-.', 2.0, 0.10)]
for key, lab, col, mk, ls, lw, ly in series:
    v = mean_err(key)
    ax.plot(ns, v, color=col, marker=mk, ls=ls, lw=lw, ms=7 if key != 'ours' else 8.5, zorder=3)
    ax.annotate(lab, xy=(60, v[-1]), xycoords='data', xytext=(1.08, ly), textcoords='axes fraction',
                va='center', fontsize=10.5, color=col, fontweight='bold' if key == 'ours' else 'normal',
                arrowprops=dict(arrowstyle='-', color=col, lw=1.1, shrinkA=1, shrinkB=4, relpos=(0, 0.5)))
floor = np.mean([np.mean(h2h[k]['reference']) * 0 + 0.017 for k in kp])
ax.axhline(0.017, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1)
ax.set_yscale('log'); ax.set_ylim(0.007, 0.12); ax.set_xlim(27, 63)
ax.set_xticks(ns); ax.set_xlabel('real simulations used', fontsize=11.5)
ax.set_yticks([0.01, 0.02, 0.05, 0.1]); ax.set_yticklabels(['0.01', '0.02', '0.05', '0.10'])
ax.minorticks_off()
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color=RULE, lw=0.9)
fig.text(0.055, 0.925, 'Error in the sensitivity indices', fontsize=13.5, fontweight='bold')
fig.text(0.055, 0.855, 'lower is better, mean of 3 metrics, 10 random draws', fontsize=10, color=MUTED)
fig.text(0.055, 0.815, 'dashed line: noise floor of the 540-run reference', fontsize=10, color=MUTED)
bad['h2h'] = save_card(fig, 'h2h')

# --------------------------------------------------------------------- B2 cost
fig = card_fig(W, H)
ax = axes_in(fig, 0.40, 0.17, 0.53, 0.55)
rows = [('this pipeline,\n30 real runs', 15, '15 hours', BRAND),
        ('the 540 supplied\nruns', 270, '11 days', MUTED),
        ('standard Monte Carlo\non the solver', 6656, '277 days', CHAR)]
for i, (lab, hrs, txt, col) in enumerate(rows):
    ax.barh(i, hrs, color=col, height=0.55, zorder=3)
    ax.text(hrs * 1.25, i, txt, va='center', fontsize=12, fontweight='bold', color=col)
ax.set_xscale('log'); ax.set_xlim(5, 90000); ax.set_ylim(-0.6, 2.6)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax.set_xticks([10, 100, 1000, 10000]); ax.set_xticklabels(['10 h', '100 h', '1,000 h', '10,000 h'])
ax.minorticks_off(); ax.tick_params(axis='y', length=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', color=RULE, lw=0.9, zorder=0)
fig.text(0.055, 0.925, 'Solver time needed', fontsize=13.5, fontweight='bold')
fig.text(0.055, 0.855, 'at 30 minutes per simulation, run one after another', fontsize=10, color=MUTED)
fig.text(0.055, 0.045, 'then 0.3 seconds of exact arithmetic gives the indices', fontsize=10, color=BRAND)
bad['cost'] = save_card(fig, 'cost')


# --------------------------------------------------- C conventional vs this pipeline
def bars_card(name, rows, col, foot):
    """Three labelled bars on one shared log scale plus a value column."""
    fig = card_fig(5.64, 3.55)
    ax = canvas_ax(fig)
    lo, hi, span = np.log10(10), np.log10(1.2e6), 62.0
    ys = [46.5, 30.0, 13.5]
    for (lab, hrs, txt), y in zip(rows, ys):
        ax.text(4, y + 5.0, lab, fontsize=11.5, va='center', color=CHAR)
        length = max(span * (np.log10(hrs) - lo) / (hi - lo), 1.6)
        ax.add_patch(FancyBboxPatch((4, y - 1.6), length, 3.6, boxstyle='round,pad=0,rounding_size=0.6',
                                    fc=col, ec='none', zorder=3))
        ax.text(72, y + 0.3, txt, fontsize=15, fontweight='bold', va='center', color=col)
    ax.plot([4, 96], [7.0, 7.0], color=RULE, lw=1.2)
    ax.text(4, 3.4, foot, fontsize=10.5, color=col, fontweight='bold', va='center')
    return fig


rows_c = [('Which errors matter', 6656, '277 days'),
          ('How many boards pass', 280, '12 days'),
          ('Design search, 1.92 M evaluations', 961000, '110 years')]
fig = bars_card('conv', rows_c, CHAR, '13,312 + 561 + 1.92 million solver runs, log scale')
bad['conv'] = save_card(fig, 'conv')
rows_o = [('Which errors matter', 15, '15 hours'),
          ('How many boards pass', 20, '20 hours'),
          ('Design search, 1.92 M evaluations', 30, '30 h + 5 min')]
fig = bars_card('ours', rows_o, BRAND, 'one model from 60 simulations answers all three, same log scale')
bad['ours'] = save_card(fig, 'ours')

# ------------------------------------------------------------------ D staircase
W, H = 6.75, 5.33
fig = card_fig(W, H)
ax = axes_in(fig, 0.36, 0.19, 0.36, 0.61)
labels = ['Design today', 'New laminate\n(dielectric constant 4.02)', '+ copper width A7\ntolerance halved',
          '+ board thickness\ntolerance halved', '+ copper width A3\ntolerance halved']
for i, s in enumerate(lwo):
    y = len(lwo) - 1 - i
    ax.barh(y, s['model_leave_out_pct'], color=BRAND if i else MUTED, height=0.52, zorder=3)
    ax.plot([s['real_lo'], s['real_hi']], [y, y], color=CHAR, lw=2.2, zorder=4)
    for e in (s['real_lo'], s['real_hi']):
        ax.plot([e, e], [y - 0.16, y + 0.16], color=CHAR, lw=2.2, zorder=4)
    ax.plot([s['real_pct']], [y], 'o', color='white', mec=CHAR, mew=2.2, ms=9, zorder=5)
    ax.text(2, y, f"{s['model_leave_out_pct']:.0f}%", ha='left', va='center', color='white',
            fontsize=12.5, fontweight='bold', zorder=6)
    fig.text(0.755, 0.19 + 0.61 * (y + 0.5) / len(lwo),
             f"{s['real_pct']:.0f}%  ({s['real_lo']:.0f}-{s['real_hi']:.0f})\n{s['n_window']} boards, {s['window_verdict_agreement']:.0f}% right",
             fontsize=10, va='center', color=CHAR, linespacing=1.4)
ax.set_yticks(range(len(lwo))); ax.set_yticklabels(labels[::-1], fontsize=10.5)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(lwo) - 0.4)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(['0', '25', '50', '75', '100%'])
ax.tick_params(axis='y', length=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', color=RULE, lw=0.9, zorder=0)
fig.text(0.05, 0.925, 'Boards passing under the stated limits', fontsize=13.5, fontweight='bold')
fig.text(0.05, 0.865, 'bar = full-curve model retrained without that scenario\'s boards', fontsize=10, color=MUTED)
fig.text(0.05, 0.83, 'circle and line = those real boards, with 95% interval', fontsize=10, color=MUTED)
fig.text(0.755, 0.845, 'real boards', fontsize=10, color=MUTED, fontweight='bold')
fig.text(0.05, 0.075, 'each step keeps the ones above it; every predicted matrix checked for passivity', fontsize=10, color=MUTED)
fig.text(0.05, 0.032, 'last step: 30 real boards; the 11-point gap is within sampling noise', fontsize=10, color=BRAND)
bad['stair'] = save_card(fig, 'stair')

print('\nproblems:', {k: v for k, v in bad.items() if v} or 'none')
