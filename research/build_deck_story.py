"""Story figures for the submission deck: the problem slide's stat tiles and the
state-of-the-art comparison. Same card style and the same no-overlap check as
build_deck_figs.py."""
import json
from matplotlib.patches import FancyBboxPatch
from deck_style import *

bad = {}
fc = json.load(open('results_fc_summary.json'))
n_mat = fc['totals']['n_matrices']

# ------------------------------------------------------------------ problem tiles
W, H = 11.5, 3.3
fig = card_fig(W, H)
ax = canvas_ax(fig)                       # 100 x 28.7 units
tiles = [('30 min', 'one full-wave simulation', 'per board design', CHAR),
         ('277 days', 'sensitivity analysis', 'solver only: 13,312 runs (KPI 2)', BRAND),
         ('110 years', 'design search', 'solver only: 1.92 M runs (KPI 1)', BRAND),
         ('0.25-0.40 dB', 'design margin today', 'vs 1.3 dB of factory spread', CHAR)]
x0, tw, gap = 3.0, 22.0, 2.0
for i, (big, l1, l2, col) in enumerate(tiles):
    x = x0 + i * (tw + gap)
    ax.add_patch(FancyBboxPatch((x, 5.2), tw, 20.0, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=SOFT, ec='none', zorder=1))
    ax.text(x + 1.8, 19.6, big, fontsize=25 if len(big) < 10 else 20, fontweight='bold', color=col, va='center')
    ax.text(x + 1.8, 13.4, l1, fontsize=11.5, color=CHAR, va='center', fontweight='bold')
    ax.text(x + 1.8, 9.4, l2, fontsize=10.5, color=MUTED, va='center')
ax.text(3.0, 2.2, 'Serial solver time at 30 minutes per run. Tolerance studies need thousands to millions of runs.',
        fontsize=10, color=MUTED, va='center')
bad['problem'] = save_card(fig, 'problem')

# ------------------------------------------------------------- state of the art
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
         'Applied to the 11 manufacturing tolerances;\ntop drivers right from 30 runs'),
        ('Yield', 'GP-assisted Monte Carlo that re-checks\nborderline boards in the solver [7]',
         'No extra runs allowed: every pass rate checked\nagainst held-out real boards')]
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
