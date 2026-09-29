import numpy as np, json, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from vis import *
from vis import _own
import load_huawei as L
P, f, S = L.load()
db = lambda x: 20*np.log10(np.clip(np.abs(x), 1e-12, None))
bad = {}

# ---- 06  one board across frequency ----------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.2))
k = 0
ax.plot(f, db(S[k, :, 1, 0]), color=BLUE, lw=3,
        label='S21   through path, port 1 to port 2')
ax.plot(f, db(S[k, :, 2, 0]), color=ORANGE, lw=3, ls='--',
        label='S31   tapped path, port 1 to port 3')
ax.plot(f, db(S[k, :, 0, 0]), color=VERM, lw=3, ls=':',
        label='S11   reflected back out of port 1')
i = int(np.argmin(db(S[k, :, 0, 0])))
ax.annotate('resonance', xy=(f[i], db(S[k, i, 0, 0])), xytext=(1.243, -26.0),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=INK, lw=2))
ax.set_xlim(1.195, 1.705)
ax.set_ylim(-30, 2)
ax.set_xlabel('frequency in GHz', fontsize=12.5)
ax.set_ylabel('decibels', fontsize=12.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color=FAINT, lw=1.4)
ax.set_title('One board, measured at all 251 frequencies', fontsize=14, loc='left', pad=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
          fontsize=11)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.26)
bad['06_sweep'] = save(fig, '06_sweep')

# ---- 07  manufacturing scatter ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.45, 1], 'wspace': 0.32})
a = axes[0]
for kk in range(0, 540, 3):
    a.plot(f, db(S[kk, :, 0, 0]), color=BLUE, lw=0.6, alpha=0.18)
a.plot(f, np.median(db(S[:, :, 0, 0]), 0), color=INK, lw=3)
a.set_xlabel('frequency in GHz', fontsize=12)
a.set_ylabel('reflection S11, dB', fontsize=12)
a.spines[['top', 'right']].set_visible(False)
a.set_title('All 540 boards', fontsize=13, loc='left', pad=12)

b = axes[1]
w = db(S[:, :, 0, 0]).max(1)
b.hist(w, bins=28, color=ORANGE)
b.set_xlabel('worst reflection in band, dB', fontsize=12)
b.set_ylabel('number of boards', fontsize=12)
b.spines[['top', 'right']].set_visible(False)
b.set_title('The spread that causes rejects', fontsize=13, loc='left', pad=12)
fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.21, wspace=0.42)
fig.text(0.07, 0.035, 'Each faint line is one manufactured board. The dark line is the typical one.',
         fontsize=10.5, color=GREY)
bad['07_spread'] = save(fig, '07_spread')

# ---- 10  which errors matter -------------------------------------------------
kk = json.load(open('results_kpi.json'))
st = np.array(kk['sobol']['S11_max_dB']['ST']); nm = kk['param_names']
o = np.argsort(st)
pretty = {'$DK': 'material constant', '$R0402_1': 'resistor 1', '$R0402_2': 'resistor 2',
          '$R0402_RL': 'load resistor', 'TOL_LW_A1': 'copper width A1',
          'TOL_LW_A3': 'copper width A3', 'TOL_LW_A5': 'copper width A5',
          'TOL_LW_A7': 'copper width A7', 'TOL_LW_A9': 'copper width A9',
          'TOL_h': 'board thickness', 't_art1': 'copper thickness'}
labs = [pretty[nm[i]] for i in o]
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.barh(labs, st[o], color=[ORANGE if v > 0.3 else GREY for v in st[o]], height=0.62)
for lab, v in zip(labs, st[o]):
    ax.text(v + 0.010, lab, f'{v:.3f}', va='center', fontsize=11.5)
ax.set_xlim(0, 0.82); ax.set_xticks([])
ax.set_xlabel('share of the variation in reflection explained by this one error', fontsize=11.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Which manufacturing errors actually matter', fontsize=14, loc='left', pad=14)
ax.text(0.62, 9.3, 'These two explain\n89% between them', fontsize=12.5, color=ORANGE,
        fontweight='bold', va='center', linespacing=1.6)
ax.text(0.62, 3.8, 'These nine barely\nregister at all', fontsize=12.5, color=GREY,
        va='center', linespacing=1.6)
fig.tight_layout()
bad['10_sensitivity'] = save(fig, '10_sensitivity')

# ---- 12  yield before and after ---------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.2))
for row, (pas, lab) in enumerate([(87.5, 'after the model redesigns it'), (39.5, 'the design today')]):
    ax.barh(row, pas, color=GREEN, height=0.46)
    ax.barh(row, 100 - pas, left=pas, color=FAINT, height=0.46)
    ax.text(pas/2, row, f'{pas}% pass', ha='center', va='center', color=WHITE,
            fontweight='bold', fontsize=13)
    ax.text(pas + (100-pas)/2, row, f'{100-pas:.1f}% rejected', ha='center', va='center',
            color=GREY, fontsize=11.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(['the design today', 'after redesign'],
                                          fontsize=12.5, fontweight='bold')
ax.set_xticks([]); ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.8)
ax.spines[:].set_visible(False)
ax.set_title('Out of every 100 boards manufactured, how many meet specification',
             fontsize=14, loc='left', pad=16)
fig.tight_layout()
bad['12_yield'] = save(fig, '12_yield')

# ---- 14  how little data is needed -------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 5.0))
n = [60, 120, 240, 400]
ax.plot(n, [0.0435, 0.0313, 0.0220, 0.0145], 'o-', color=GREY, lw=2.4, ms=8)
ax.plot([60, 120], [0.0160, 0.0148], 's-', color=ORANGE, lw=3.4, ms=10)
ax.text(78, 0.0470, 'with my compression step', color=GREY, fontsize=12, fontweight='bold')
ax.text(132, 0.0150, 'without it, the right way', color=ORANGE, fontsize=12,
        fontweight='bold', va='center')
ax.text(132, 0.0131, '60 designs here beat 400 designs there', color=ORANGE,
        fontsize=12, fontweight='bold', va='center')
ax.set_yscale('log'); ax.set_xlim(40, 445); ax.set_ylim(0.0120, 0.062)
ax.set_xlabel('number of simulated designs used for training', fontsize=12.5)
ax.set_ylabel('prediction error', fontsize=12.5)
ax.set_yticks([0.015, 0.02, 0.03, 0.05]); ax.set_yticklabels(['1.5%', '2%', '3%', '5%'])
ax.minorticks_off()
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('You do not need all 540 simulations', fontsize=14, loc='left', pad=14)
fig.tight_layout()
bad['14_dataeff'] = save(fig, '14_dataeff')

print('\nproblems:', {k: v for k, v in bad.items() if v})
