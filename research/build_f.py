import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from vis import *
bad = {}

# ---- 19  which model family, on your data -----------------------------------
r = pd.DataFrame(json.load(open('results_classical.json')))
m = r[r.n_train == 400].groupby('model').rel_L2.mean()
names = ['Kriging, uncompressed', 'Kriging', 'Gradient boosting', 'Linear ridge',
         'Neural network', 'Quadratic']
vals = [0.0148, float(m['Kriging (GP, ARD Matern)']), float(m['Grad. boosting']),
        float(m['Ridge (linear)']), 0.0209, float(m['Poly2 + Ridge'])]
cols = [ORANGE] + [GREY] * 5
order = np.argsort(vals)[::-1]
fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.barh([names[i] for i in order], [vals[i] * 100 for i in order],
        color=[cols[i] for i in order], height=0.6)
for i in order:
    ax.text(vals[i] * 100 + 0.06, names[i], '%.2f%%' % (vals[i] * 100),
            va='center', fontsize=11.5)
ax.set_xlim(0, 3.8)
ax.set_xticks([])
ax.set_xlabel('prediction error on 140 held-out designs, lower is better', fontsize=11.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Six model families, tested on your data under identical conditions',
             fontsize=14, loc='left', pad=14)
ax.text(2.25, 4.6, 'the winner, and it\nneeds only 60 designs',
        fontsize=11.5, color=ORANGE, fontweight='bold', va='center', linespacing=1.6)
fig.tight_layout()
bad['19_models'] = save(fig, '19_models')

# ---- 20  what happens next ---------------------------------------------------
fig, ax = canvas(13, 5.6)
ax.text(3, 41, 'What to do next, in order', fontsize=15.5, fontweight='bold')
steps = [('NOW', 'Send the six\nquestions', 'nothing else\nis blocked by it', VERM),
         ('WEEK 1', 'Rebuild on\nuncompressed kriging', 'the benchmark\nalready says so', ORANGE),
         ('WEEK 2', 'Add an active\nlearning loop', 'prove how few\nruns you need', BLUE),
         ('WEEK 3', 'Confirm the redesign\nin the real solver', 'turns a prediction\ninto a result', GREEN),
         ('WEEK 4', 'Write it around\nthe two ratios', 'the criteria are\nthe outline', INK)]
for i, (when, what, why, col) in enumerate(steps):
    x = 2 + i * 19.4
    ax.add_patch(Circle((x + 8, 29), 2.4, fc=col, ec='none', zorder=3))
    ax.text(x + 8, 29, str(i + 1), ha='center', va='center', color=WHITE,
            fontsize=12, fontweight='bold', zorder=4)
    if i < 4:
        ax.plot([x + 10.8, x + 24.6], [29, 29], color=FAINT, lw=3, zorder=1)
    ax.text(x + 8, 23.5, when, ha='center', fontsize=10, fontweight='bold', color=col)
    ax.text(x + 8, 17.5, what, ha='center', fontsize=10.6, fontweight='bold',
            va='center', linespacing=1.6)
    ax.text(x + 8, 9.0, why, ha='center', fontsize=9.6, color=GREY,
            va='center', linespacing=1.6)
bad['20_plan'] = save(fig, '20_plan')

print('problems:', {k: v for k, v in bad.items() if v})
