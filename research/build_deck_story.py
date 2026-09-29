"""Story figures for the submission deck: problem tiles (slide 2), how-it-works flow
(slide 3), sensitivity comparison (slide 4), design-search comparison (slide 5) and
the state-of-the-art table (slide 7). Same card style and the same no-overlap check
as build_deck_figs.py. Orange is reserved for this project; the conventional route
and neutral facts are charcoal."""
import json
import numpy as np
from matplotlib.patches import FancyBboxPatch
from deck_style import *

bad = {}
fc = json.load(open('results_fc_summary.json'))
h2h = json.load(open('results_head2head.json'))
n_mat = fc['totals']['n_matrices']


def box(ax, x, y, w, h, fc=SOFT, ec='none', lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=lw, zorder=1))


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=2.0, mutation_scale=16, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ 2 problem tiles
W, H = 11.5, 3.3
fig = card_fig(W, H)
ax = canvas_ax(fig)                       # 100 x 28.7 units
tiles = [('30 min', 'one full-wave simulation', 'per board design'),
         ('277 days', 'sensitivity analysis', 'solver only: 13,312 simulations'),
         ('110 years', 'design search', 'solver only: 1.92 M simulations'),
         ('30%', 'of boards pass today', '0.3 dB margin vs 1.3 dB factory spread')]
x0, tw, gap = 3.0, 22.0, 2.0
for i, (big, l1, l2) in enumerate(tiles):
    x = x0 + i * (tw + gap)
    box(ax, x, 5.2, tw, 20.0)
    ax.text(x + 1.8, 19.6, big, fontsize=26, fontweight='bold', color=CHAR, va='center')
    ax.text(x + 1.8, 13.4, l1, fontsize=11.5, color=CHAR, va='center', fontweight='bold')
    ax.text(x + 1.8, 9.4, l2, fontsize=10, color=MUTED, va='center')
ax.text(3.0, 2.2, 'Serial solver time at 30 minutes per simulation. Pass rate under illustrative acceptance limits.',
        fontsize=10, color=MUTED, va='center')
bad['problem'] = save_card(fig, 'problem')

# ------------------------------------------------------------------ 3 how it works
W, H = 5.67, 4.76
fig = card_fig(W, H)
ax = canvas_ax(fig)                       # 100 x 84 units
ax.text(5, 78.5, 'How it works', fontsize=13.5, fontweight='bold', color=CHAR, va='center')
box(ax, 5, 60, 90, 12)
ax.text(9, 68.4, '60 full-wave simulations', fontsize=12.5, fontweight='bold', color=CHAR, va='center')
ax.text(9, 63.6, '11 manufacturing tolerances in, S-parameters out', fontsize=10.5, color=MUTED, va='center')
arrow(ax, 50, 59.4, 50, 54.6)
box(ax, 5, 34, 90, 20, fc='white', ec=BRAND, lw=2.2)
ax.text(9, 48.6, 'AI model (kriging)', fontsize=12.5, fontweight='bold', color=BRAND, va='center')
ax.text(9, 43.8, 'predicts every S-parameter in milliseconds', fontsize=10.5, color=CHAR, va='center')
ax.text(9, 38.6, 'physics: reciprocity built in, passivity checked', fontsize=10.5, color=CHAR, va='center')
arrow(ax, 27, 33.4, 27, 28.6)
arrow(ax, 73, 33.4, 73, 28.6)
for x, t1, t2, tag in ((5, 'Sensitivity analysis', 'which tolerances matter', 'KPI 2'),
                       (52, 'Design search', 'which design passes most', 'KPI 1')):
    box(ax, x, 8, 43, 20)
    ax.text(x + 3.5, 22.0, t1, fontsize=11.5, fontweight='bold', color=CHAR, va='center')
    ax.text(x + 3.5, 17.2, t2, fontsize=10, color=MUTED, va='center')
    ax.add_patch(FancyBboxPatch((x + 3.5, 10.0), 11, 3.4, boxstyle='round,pad=0,rounding_size=1.4',
                                fc=BRAND, ec='none', zorder=2))
    ax.text(x + 9.0, 11.7, tag, fontsize=9.5, fontweight='bold', color='white', ha='center', va='center', zorder=3)
bad['flow'] = save_card(fig, 'flow')

# ------------------------------------------------------------------ 4 sensitivity accuracy
W, H = 5.67, 4.25
kp = ['S11_max_dB', 'S31_mean_dB', 'iso_max_dB']
err = lambda m: float(np.mean([h2h[k]['n30'][m]['mad'] for k in kp]))
conv, ours = err('binned'), err('ours')
fig = card_fig(W, H)
ax = axes_in(fig, 0.36, 0.30, 0.52, 0.38)
rows = [('This project\n(kriging)', ours, BRAND), ('Conventional\nMonte Carlo', conv, CHAR)]
for i, (lab, v, col) in enumerate(rows):
    ax.barh(i, v, color=col, height=0.55, zorder=3)
    ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=12.5, fontweight='bold', color=col)
ax.set_yticks(range(2)); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlim(0, 0.1); ax.set_ylim(-0.6, 1.6)
ax.set_xticks([0, 0.05, 0.1]); ax.set_xticklabels(['0', '0.05', '0.10'])
ax.tick_params(axis='y', length=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', color=RULE, lw=0.9, zorder=0)
ax.set_xlabel('error of the sensitivity indices', fontsize=10.5)
fig.text(0.055, 0.915, 'Sensitivity analysis (KPI 2)', fontsize=13.5, fontweight='bold')
fig.text(0.055, 0.845, 'both from the same 30 simulations; lower is better', fontsize=10, color=MUTED)
fig.text(0.055, 0.765, f'{conv / ours:.0f}x lower error', fontsize=15, fontweight='bold', color=BRAND)
fig.text(0.055, 0.075, 'Polynomial chaos and a neural network reach the same accuracy as kriging.',
         fontsize=9.5, color=MUTED)
bad['sens'] = save_card(fig, 'sens')

# ------------------------------------------------------------------ 5 design search
W, H = 11.5, 3.9
fig = card_fig(W, H)
ax = canvas_ax(fig)                       # 100 x 33.9 units
ax.text(3, 30.4, 'Design search (KPI 1): 961 candidate designs, each tested on 2,000 virtual boards',
        fontsize=12.5, fontweight='bold', color=CHAR, va='center')
for x, head, big, sub, col in ((3, 'Solver only', '110 years', '1.92 million simulations, one after another', CHAR),
                               (52, 'This project', '30 hours', '60 simulations to train, then 5 minutes of computing', BRAND)):
    box(ax, x, 6.5, 45, 19.5, fc=SOFT if col == CHAR else 'white', ec='none' if col == CHAR else BRAND,
        lw=0 if col == CHAR else 2.2)
    ax.text(x + 2.5, 22.2, head, fontsize=12, fontweight='bold', color=col, va='center')
    ax.text(x + 2.5, 15.0, big, fontsize=34, fontweight='bold', color=col, va='center')
    ax.text(x + 2.5, 9.2, sub, fontsize=10.5, color=MUTED, va='center')
ax.text(3, 2.6, 'Serial time at 30 minutes per simulation. With 100 solver licences in parallel, the solver-only search '
        'still takes about a year.', fontsize=10, color=MUTED, va='center')
bad['design'] = save_card(fig, 'design')

# ------------------------------------------------------------------ 7 state of the art
W, H = 11.5, 3.55
fig = card_fig(W, H)
ax = canvas_ax(fig)                       # 100 x 30.9 units
cols = [(3.0, ''), (17.0, 'Common practice in the literature'), (58.0, 'This project')]
top = 27.6
for x, head in cols:
    ax.text(x, top, head, fontsize=11.5, fontweight='bold', va='center',
            color=BRAND if head == 'This project' else CHAR)
rows = [('Surrogate', 'Kriging / Gaussian process, about a third of\nRF surrogate studies [4]',
         'Kriging, checked against polynomial chaos and\na neural network on the same boards'),
        ('Output', 'Compress the frequency response first\n(PCA or response features)',
         'No compression: PCA raised the error 2.8x on\nthis data, so all 3,012 outputs are predicted'),
        ('Physics', "Physics-informed networks put Maxwell's\nequations in the loss; they need field data [6]",
         f'Only port data exists: reciprocity built in,\npassivity checked on {n_mat / 1e6:.0f} million matrices'),
        ('Sensitivity', 'Closed-form Sobol indices from a GP [1-3];\nkriging-PCE in EM dosimetry [5]',
         'Applied to the 11 manufacturing tolerances;\nleading tolerances found from 30 simulations'),
        ('Pass rate', 'GP-assisted Monte Carlo that re-checks\nborderline boards in the solver [7]',
         'No extra simulations allowed: every pass rate\nchecked against held-out real boards')]
rh = 4.7
ax.plot([3, 97], [top - 2.2, top - 2.2], color=RULE, lw=1.4)
for i, (lab, a, b) in enumerate(rows):
    y = top - 2.2 - rh * (i + 0.5)
    if i:
        ax.plot([3, 97], [y + rh / 2, y + rh / 2], color=RULE, lw=0.8)
    ax.text(3.0, y, lab, fontsize=10.5, fontweight='bold', va='center', color=CHAR)
    ax.text(17.0, y, a, fontsize=9.6, va='center', color=MUTED, linespacing=1.3)
    ax.text(58.0, y, b, fontsize=9.6, va='center', color=CHAR, linespacing=1.3)
bad['sota'] = save_card(fig, 'sota')

print('problems:', {k: v for k, v in bad.items() if v} or 'none')
