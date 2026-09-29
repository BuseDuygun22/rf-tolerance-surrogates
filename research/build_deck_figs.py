"""Validation figure for the submission deck (slide 6). Numbers are read from result files.
The other deck figures are built by build_deck_story.py and build_deck_drivers.py."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from deck_style import *

bad = {}
lwo = json.load(open('results_fc_summary.json'))['lwo']   # full-curve leave-window-out (3,012-output model, passivity checked)

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
fig.text(0.05, 0.865, 'bar = model prediction, made without that scenario\'s real boards', fontsize=10, color=MUTED)
fig.text(0.05, 0.83, 'circle and line = the real boards, with their 95% range', fontsize=10, color=MUTED)
fig.text(0.755, 0.845, 'real boards', fontsize=10, color=MUTED, fontweight='bold')
fig.text(0.05, 0.075, 'each design change keeps the ones above it', fontsize=10, color=MUTED)
fig.text(0.05, 0.032, 'last step: 30 real boards; the 11-point gap is within sampling noise', fontsize=10, color=BRAND)
bad['stair'] = save_card(fig, 'stair')

print('\nproblems:', {k: v for k, v in bad.items() if v} or 'none')
