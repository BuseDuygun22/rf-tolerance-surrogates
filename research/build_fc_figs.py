"""Frequency-resolved failure map from the full-curve model, for the project folder."""
import json
import numpy as np
import matplotlib.pyplot as plt
from deck_style import *
import load_huawei as L

st = json.load(open('fc_stair.json'))
f = np.array(st['freq'])
m0, m1 = st['maps']['today'], st['maps']['+A7+h+A3']

fig = card_fig(11.0, 4.6)
axs = [axes_in(fig, 0.07, 0.20, 0.40, 0.55), axes_in(fig, 0.56, 0.20, 0.40, 0.55)]
for ax, key, title, limit in ((axs[0], 'viol11', 'Reflection above the limit', '-5.4 dB'),
                              (axs[1], 'violiso', 'Isolation above the limit', '-14.9 dB')):
    ax.plot(f, m0[key], color=MUTED, lw=2.8, ls='--')
    ax.plot(f, m1[key], color=BRAND, lw=3.2)
    ax.set_xlim(f[0] - 0.012, f[-1]); ax.set_ylim(0, max(max(m0[key]), max(m1[key])) * 1.25)
    ax.set_xlabel('frequency in GHz', fontsize=11)
    ax.set_ylabel('boards failing at that frequency (%)', fontsize=10.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=RULE, lw=0.9)
    ax.tick_params(axis='y', pad=8)
    ax.tick_params(axis='x', pad=6)
    ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left', pad=8)
    ax.text(0.5, 0.93, f'limit {limit}', transform=ax.transAxes, fontsize=10, color=MUTED, ha='center')
    ymax = ax.get_ylim()[1]
    ax.text(f[0] + 0.012, m0[key][2] + ymax * 0.05, 'design today', fontsize=10.5, color=MUTED, fontweight='bold')
    ax.text(f[-1] - 0.012, m1[key][-3] + ymax * 0.06, 'after the full design', fontsize=10.5, color=BRAND,
            fontweight='bold', ha='right')
fig.text(0.035, 0.925, 'Where in the band boards fail, from the full-curve model', fontsize=14, fontweight='bold')
fig.text(0.035, 0.865, '100,000 virtual boards per design, every predicted matrix checked for passivity',
         fontsize=10.5, color=MUTED)
problems = save_card(fig, 'freqmap')
print('problems:', problems or 'none')

edge = 0.05
band = np.array(f)
for key, name in (('viol11', 'reflection'), ('violiso', 'isolation')):
    for tag, m in (('today', m0), ('full design', m1)):
        v = np.array(m[key])
        share = v[(band <= band[0] + edge) | (band >= band[-1] - edge)].sum() / v.sum() if v.sum() else float('nan')
        peak = band[int(np.argmax(v))]
        print(f'{name:10s} {tag:12s} peak violation {v.max():5.1f}% at {peak:.3f} GHz; share within 50 MHz of a band edge {100*share:4.0f}%')
