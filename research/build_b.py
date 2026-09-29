import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from vis import *
from vis import _own
bad = {}

# ---- 04  the S-parameter grid ----------------------------------------------
fig, ax = canvas(13, 7.0)
ax.text(3, 49, 'Nine measurements, but only six are independent',
        fontsize=15.5, fontweight='bold')
x0, y0, c, g = 5, 5, 11, 1.4
for i in range(3):
    for j in range(3):
        x, y = x0 + j*(c+g), y0 + (2-i)*(c+g)
        mir = i > j
        ax.add_patch(Rectangle((x, y), c, c, fc=FAINT if mir else WHITE,
                               ec=GREY if mir else INK, lw=1.2 if mir else 2.2, zorder=2))
        t1 = ax.text(x + c/2, y + c*0.63, f'S{i+1}{j+1}', ha='center', fontsize=14,
                     fontweight='bold', color=GREY if mir else INK, zorder=3)
        lab = 'reflects' if i == j else f'{j+1} to {i+1}'
        t2 = ax.text(x + c/2, y + c*0.28, lab, ha='center', fontsize=9.5,
                     color=GREY if mir else INK, zorder=3)
        _own(fig, t1, t2)
ax.text(x0, 43.5, 'goes INTO port   1            2            3',
        fontsize=10.5, color=GREY)
ax.text(x0 - 2.5, y0 + 1.5*(c+g), 'comes OUT of port', fontsize=10.5, color=GREY,
        rotation=90, va='center', ha='center')
ax.text(x0, 1.0, 'S12 = S21          S13 = S31          S23 = S32',
        fontsize=13, fontweight='bold', color=ORANGE)
px = 49
ax.text(px, 41, 'RECIPROCITY', fontsize=13, fontweight='bold', color=ORANGE)
ax.text(px, 36.5,
        'A component with no power supply passes a\n'
        'signal equally well in either direction.\n\n'
        'So the shaded cells are copies of the white\n'
        'ones directly opposite them. This holds in\n'
        'your data to eleven decimal places.',
        fontsize=11.5, va='top', linespacing=1.7)
ax.text(px, 14.0, 'The model predicts 6 numbers.\nThe other 3 are copied for free.',
        fontsize=12, va='top', color=ORANGE, fontweight='bold', linespacing=1.7)
bad['04_sparams'] = save(fig, '04_sparams')

# ---- 08  slow way, fast way -------------------------------------------------
fig, ax = canvas(13, 5.8)
ax.text(3, 43, 'Why a simulation is slow, and why the model is not',
        fontsize=15.5, fontweight='bold')
ax.text(3, 35.5, 'THE SIMULATOR', fontsize=11, fontweight='bold', color=VERM)
steps = [('geometry', 'and materials'), ('Maxwell', 'equations'),
         ('fields', 'everywhere'), ('S-parameters', 'the answer')]
for i, (t_, s_) in enumerate(steps):
    card(ax, 3 + i*22, 24, 17, 9, t_, s_, ec=VERM if i < 3 else INK,
         title_size=12, sub_size=9.5)
    if i < 3:
        link(ax, 20 + i*22, 28.5, 25 + i*22, 28.5, VERM, 2.2)
ax.text(92, 28.5, 'minutes\neach', fontsize=12, color=VERM, fontweight='bold',
        va='center', linespacing=1.6)
ax.text(3, 17, 'THE TRAINED MODEL', fontsize=11, fontweight='bold', color=GREEN)
for i, (t_, s_) in enumerate([('geometry', 'and materials'), ('the model', 'learned pattern'),
                              ('S-parameters', 'the answer')]):
    card(ax, 3 + i*22, 5.5, 17, 9, t_, s_, ec=GREEN if i < 2 else INK,
         title_size=12, sub_size=9.5)
    if i < 2:
        link(ax, 20 + i*22, 10, 25 + i*22, 10, GREEN, 2.2)
ax.text(70, 10, '0.03 milliseconds each', fontsize=12, color=GREEN,
        fontweight='bold', va='center')
bad['08_slow_fast'] = save(fig, '08_slow_fast')

# ---- 09  the architecture ----------------------------------------------------
fig, ax = canvas(13, 8.0)
ax.text(3, 58, 'How the whole thing fits together', fontsize=15.5, fontweight='bold')
ax.text(3, 52, 'STEP 1   PAID ONCE', fontsize=10.5, fontweight='bold', color=GREY)
card(ax, 3, 41, 20, 8.5, '60 designs', 'simulated in CST', fc=FAINT, ec=INK,
     title_size=12, sub_size=9.5)
link(ax, 23, 45.2, 27, 45.2, INK, 2.2)
card(ax, 27, 41, 22, 8.5, 'TRAIN THE MODEL', 'about five minutes', ec=BLUE,
     title_size=12, sub_size=9.5)
link(ax, 49, 45.2, 53, 45.2, INK, 2.2)
card(ax, 53, 39.5, 27, 11.5, 'THE FAST MODEL', 'answers in 0.03 ms', ec=BLUE, lw=3.2,
     title_size=13.5, sub_size=10.5)
ax.text(3, 36.5, 'STEP 2   PHYSICS BUILT IN', fontsize=10.5, fontweight='bold', color=ORANGE)
for i, (t_, s_) in enumerate([('reciprocity', 'six outputs, not nine'),
                              ('passivity', 'checked, not forced'),
                              ('smoothness', 'why 60 is enough')]):
    card(ax, 3 + i*17, 25, 15, 8, t_, s_, fc=TINT_O, ec=ORANGE,
         title_size=11, sub_size=8.6)
    ax.plot([10.5 + i*17, 10.5 + i*17], [33, 35.2], color=ORANGE, lw=2, ls='--', zorder=1)
ax.plot([10.5, 44.5], [35.2, 35.2], color=ORANGE, lw=2, ls='--', zorder=1)
link(ax, 38, 35.2, 38, 40.6, ORANGE, 2.2, dashed=True)
ax.text(3, 21.5, 'STEP 3   RUN IT MILLIONS OF TIMES', fontsize=10.5,
        fontweight='bold', color=GREY)
for t_, s_, col, xc in [('WHICH ERRORS MATTER', '53,000 runs, 1.7 s', GREEN, 20),
                        ('HOW MANY BOARDS PASS', '100,000 runs, 3.2 s', GREEN, 50),
                        ('REDESIGN FOR YIELD', '8,000,000 runs, 150 s', ORANGE, 80)]:
    elbow(ax, 66.5, 39.5, xc, 12, col, 2.2, bus=18)
    card(ax, xc - 14, 3, 28, 9, t_, s_, ec=col, title_size=11.5, sub_size=9.5)
bad['09_architecture'] = save(fig, '09_architecture')

# ---- 11  three laws ----------------------------------------------------------
fig, ax = canvas(13, 5.8)
ax.text(3, 43, 'Three laws your measurements already obey', fontsize=15.5, fontweight='bold')
laws = [('RECIPROCITY', 'S12 = S21',
         'It passes a signal equally\nwell in either direction.',
         'BUILT IN.\nSix outputs, not nine.', GREEN),
        ('PASSIVITY', 'out  ≤  in',
         'It cannot emit more energy\nthan it receives.',
         'CHECKED, NOT FORCED.\nNever once broken.', BLUE),
        ('SMOOTHNESS', 'one resonance',
         'Behaviour changes gradually\nacross the frequency band.',
         'WHY 60 EXAMPLES\nare enough to learn from.', ORANGE)]
for i, (n_, f_, d_, u_, col) in enumerate(laws):
    x = 3 + i*31.5
    ax.add_patch(FancyBboxPatch((x, 3), 29, 33, boxstyle='round,pad=0.6,rounding_size=1.2',
                                fc=WHITE, ec=col, lw=2.6))
    ax.text(x + 14.5, 32, n_, ha='center', fontsize=12.5, fontweight='bold', color=col)
    ax.text(x + 14.5, 27, f_, ha='center', fontsize=14, fontweight='bold')
    ax.text(x + 14.5, 20.5, d_, ha='center', fontsize=11, va='center', linespacing=1.7)
    ax.plot([x + 3, x + 26], [14.5, 14.5], color=FAINT, lw=2.5)
    ax.text(x + 14.5, 9, u_, ha='center', fontsize=11, va='center', color=col,
            fontweight='bold', linespacing=1.7)
bad['11_laws'] = save(fig, '11_laws')

print('\nproblems:', {k: v for k, v in bad.items() if v})
