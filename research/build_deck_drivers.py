"""Slide 3, right panel: leading sensitivity drivers with bootstrap intervals."""
import json
import numpy as np
from deck_style import *

d = json.load(open('results_sens_intervals.json'))
PRETTY = {'$DK': 'material constant', 'TOL_h': 'board thickness', 'TOL_LW_A7': 'copper width A7',
          'TOL_LW_A3': 'copper width A3', 'TOL_LW_A9': 'copper width A9', 't_art1': 'copper thickness',
          '$R0402_RL': 'load resistor', '$R0402_1': 'resistor 1', '$R0402_2': 'resistor 2',
          'TOL_LW_A1': 'copper width A1', 'TOL_LW_A5': 'copper width A5'}
HEAD = {'S11_max_dB': 'Reflection', 'S31_mean_dB': 'Coupling', 'iso_max_dB': 'Isolation'}

W, H = 5.67, 4.25
fig = card_fig(W, H)
ax = axes_in(fig, 0.37, 0.22, 0.58, 0.55)
ytick, ylab = [], []
y = 12.6
for kpi, rows in d.items():
    ax.text(-0.62, y, HEAD[kpi], transform=ax.get_yaxis_transform(), fontsize=11.5,
            fontweight='bold', va='center', ha='left', color=CHAR)
    y -= 1.0
    for r in rows:
        ax.plot([r['lo90'], r['hi90']], [y, y], color=BRAND, lw=3.4, solid_capstyle='butt', zorder=3)
        ax.plot([r['est45']], [y], 'o', color=BRAND, ms=8, zorder=4)
        ax.plot([r['ref540']], [y - 0.34], 'D', color='white', mec=CHAR, mew=1.8, ms=7, zorder=5)
        ytick.append(y); ylab.append(PRETTY.get(r['param'], r['param']))
        y -= 1.0
    y -= 0.35
ax.set_yticks(ytick); ax.set_yticklabels(ylab, fontsize=10.5)
ax.set_ylim(y + 0.6, 13.2)
ax.set_xlim(-0.02, 0.72)
ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xticklabels(['0', '0.2', '0.4', '0.6'])
ax.tick_params(axis='y', length=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', color=RULE, lw=0.9, zorder=0)
ax.set_xlabel('share of the variation explained', fontsize=10.5)
fig.text(0.055, 0.925, 'Which tolerances matter, from 45 simulations', fontsize=13.5, fontweight='bold')
fig.text(0.055, 0.86, 'circle and line: estimate with 90% bootstrap interval', fontsize=10, color=MUTED)
fig.text(0.055, 0.82, 'diamond: the same analysis on all 540 runs', fontsize=10, color=MUTED)
fig.text(0.055, 0.03, 'three leading drivers per metric, Huawei tolerances, uniform scatter', fontsize=10, color=MUTED)
problems = save_card(fig, 'drivers')
print('problems:', problems or 'none')
print('intervals that contain the 540-run value:',
      sum(r['covered'] for rows in d.values() for r in rows), 'of', sum(len(v) for v in d.values()))
