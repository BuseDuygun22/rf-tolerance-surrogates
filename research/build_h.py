"""Figures rebuilt after Huawei's answers (2026-09-17): real 30-minute solver
cost, confirmation that S-parameter physics counts as physics-informed, and
realistic manufacturing tolerances. Every figure passes vis.check()."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from vis import *
import load_huawei as L

S_ = json.load(open('results_tol_studies.json'))
R_ = json.load(open('results_tol_redesign.json'))
ST_ = json.load(open('results_tol_staircase.json'))
bad = {}
PRETTY = {'$DK': 'material constant', '$R0402_1': 'resistor 1', '$R0402_2': 'resistor 2',
          '$R0402_RL': 'load resistor', 'TOL_LW_A1': 'copper width A1',
          'TOL_LW_A3': 'copper width A3', 'TOL_LW_A5': 'copper width A5',
          'TOL_LW_A7': 'copper width A7', 'TOL_LW_A9': 'copper width A9',
          'TOL_h': 'board thickness', 't_art1': 'copper thickness'}

# ---- 08  slow way, fast way (real solver cost) ------------------------------
fig, ax = canvas(13, 5.8)
ax.text(3, 43, 'Why a simulation is slow, and why the model is not',
        fontsize=15.5, fontweight='bold')
ax.text(3, 35.5, 'THE SIMULATOR', fontsize=11, fontweight='bold', color=VERM)
for i, (t_, s_) in enumerate([('geometry', 'and materials'), ('Maxwell', 'equations'),
                              ('fields', 'everywhere'), ('S-parameters', 'the answer')]):
    card(ax, 3 + i*22, 24, 17, 9, t_, s_, ec=VERM if i < 3 else INK, title_size=12, sub_size=9.5)
    if i < 3:
        link(ax, 20 + i*22, 28.5, 25 + i*22, 28.5, VERM, 2.2)
ax.text(91.5, 28.5, '30 min\neach', fontsize=12, color=VERM, fontweight='bold',
        va='center', linespacing=1.6)
ax.text(3, 17, 'THE TRAINED MODEL', fontsize=11, fontweight='bold', color=GREEN)
for i, (t_, s_) in enumerate([('geometry', 'and materials'), ('the model', 'learned pattern'),
                              ('S-parameters', 'the answer')]):
    card(ax, 3 + i*22, 5.5, 17, 9, t_, s_, ec=GREEN if i < 2 else INK, title_size=12, sub_size=9.5)
    if i < 2:
        link(ax, 20 + i*22, 10, 25 + i*22, 10, GREEN, 2.2)
ax.text(70, 10, 'about 0.1 milliseconds each', fontsize=12, color=GREEN,
        fontweight='bold', va='center')
bad['08_slow_fast'] = save(fig, '08_slow_fast')

# ---- 09  architecture (real timings) ----------------------------------------
fig, ax = canvas(13, 8.0)
ax.text(3, 58, 'How the whole thing fits together', fontsize=15.5, fontweight='bold')
ax.text(3, 52, 'STEP 1   PAID ONCE', fontsize=10.5, fontweight='bold', color=GREY)
card(ax, 3, 41, 20, 8.5, '60 designs', '30 solver-hours', fc=FAINT, ec=INK,
     title_size=12, sub_size=9.5)
link(ax, 23, 45.2, 27, 45.2, INK, 2.2)
card(ax, 27, 41, 22, 8.5, 'TRAIN THE MODEL', 'about four minutes', ec=BLUE,
     title_size=12, sub_size=9.5)
link(ax, 49, 45.2, 53, 45.2, INK, 2.2)
card(ax, 53, 39.5, 27, 11.5, 'THE FAST MODEL', 'about 0.1 ms per design', ec=BLUE, lw=3.2,
     title_size=13.5, sub_size=10.5)
ax.text(3, 36.5, 'STEP 2   PHYSICS BUILT IN', fontsize=10.5, fontweight='bold', color=ORANGE)
for i, (t_, s_) in enumerate([('reciprocity', 'six outputs, not nine'),
                              ('passivity', 'checked, not forced'),
                              ('smoothness', 'why 60 is enough')]):
    card(ax, 3 + i*17, 25, 15, 8, t_, s_, fc=TINT_O, ec=ORANGE, title_size=11, sub_size=8.6)
    ax.plot([10.5 + i*17, 10.5 + i*17], [33, 35.2], color=ORANGE, lw=2, ls='--', zorder=1)
ax.plot([10.5, 44.5], [35.2, 35.2], color=ORANGE, lw=2, ls='--', zorder=1)
link(ax, 38, 35.2, 38, 40.6, ORANGE, 2.2, dashed=True)
ax.text(3, 21.5, 'STEP 3   RUN IT MILLIONS OF TIMES', fontsize=10.5,
        fontweight='bold', color=GREY)
for t_, s_, col, xc in [('WHICH ERRORS MATTER', '53,248 runs, 11 s', GREEN, 20),
                        ('HOW MANY BOARDS PASS', '100,000 runs, 16 s', GREEN, 50),
                        ('REDESIGN FOR YIELD', '1.9 million runs, 5 min', ORANGE, 80)]:
    elbow(ax, 66.5, 39.5, xc, 12, col, 2.2, bus=18)
    card(ax, xc - 14, 3, 28, 9, t_, s_, ec=col, title_size=11.5, sub_size=9.5)
bad['09_architecture'] = save(fig, '09_architecture')

# ---- 10  sensitivity under real tolerances ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
panels = [('S11_max_dB', 'Reflection'), ('S31_mean_dB', 'Coupling'), ('iso_max_dB', 'Isolation')]
for ax, (k, title) in zip(axes, panels):
    st = np.array(S_['sobol_A_realistic']['indices'][k]['ST'])
    o = np.argsort(st)[::-1][:5][::-1]
    vals = st[o]
    cols = [ORANGE if v > 0.3 else (BLUE if v > 0.05 else GREY) for v in vals]
    ax.barh([PRETTY[L.PARAMS[i]] for i in o], vals, color=cols, height=0.62)
    for j, v in enumerate(vals):
        ax.text(v + 0.02, j, f'{v:.2f}', va='center', fontsize=11)
    ax.set_xlim(0, 0.82); ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=10)
    ax.tick_params(axis='y', labelsize=11)
fig.suptitle('Which manufacturing errors matter, under the real factory tolerances',
             fontsize=14.5, x=0.02, ha='left', y=0.99)
fig.text(0.02, 0.035, 'Share of each metric\'s variation explained by one parameter. '
         'Resistors and copper thickness show no measurable effect on reflection or coupling.',
         fontsize=10.5, color=GREY)
fig.subplots_adjust(left=0.115, right=0.99, top=0.84, bottom=0.12, wspace=0.95)
bad['10_sensitivity'] = save(fig, '10_sensitivity')

# ---- 12  yield staircase ------------------------------------------------------
labels = ['the design today', 'laminate changed to DK 4.02', '+ copper width A7 tolerance halved',
          '+ board thickness tolerance halved', '+ copper width A3 tolerance halved']
ys = [s['yield_pct'] for s in ST_]
fig, ax = plt.subplots(figsize=(12.5, 5.0))
rows = np.arange(len(ys))[::-1]
for r, y in zip(rows, ys):
    ax.barh(r, y, color=GREEN if r != rows[0] else GREY, height=0.58)
    ax.barh(r, 100 - y, left=y, color=FAINT, height=0.58)
    ax.text(y - 1.2, r, f'{y:.0f}%', ha='right', va='center', color=WHITE,
            fontsize=12.5, fontweight='bold')
ax.set_yticks(rows); ax.set_yticklabels(labels, fontsize=11.5)
ax.set_xlim(0, 100); ax.set_xticks([])
ax.spines[:].set_visible(False)
ax.set_title('Out of every 100 boards manufactured, how many meet specification',
             fontsize=14, loc='left', pad=14)
fig.text(0.30, 0.03, 'Each step keeps the one above it. Predictions from the model, '
         'to be confirmed in the solver.', fontsize=10.5, color=GREY)
fig.subplots_adjust(left=0.30, right=0.98, top=0.87, bottom=0.12)
bad['12_yield'] = save(fig, '12_yield')

# ---- 22  the laminate lever -------------------------------------------------
Y = np.array(R_['yield_grid']); dk = np.array(R_['grid_dk']); tg = np.array(R_['grid_t_art1'])
ti = int(np.argmin(abs(tg - R_['nominal']['t_art1'])))
fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.axvspan(dk[-1], 4.12, color=FAINT, zorder=0)
ax.plot(dk, Y[ti] * 100, color=ORANGE, lw=3.2, zorder=3)
yn = float(np.interp(R_['nominal']['dk'], dk, Y[ti] * 100))
ax.plot([R_['nominal']['dk']], [yn], 'o', color=INK, ms=10, zorder=4)
ax.plot([dk[-1]], [Y[ti, -1] * 100], 's', color=ORANGE, ms=11, zorder=4)
ax.text(R_['nominal']['dk'] + 0.004, yn - 5.5, 'today', fontsize=12, fontweight='bold')
ax.text(dk[-1] - 0.004, Y[ti, -1] * 100 + 3.5, 'chosen', fontsize=12, fontweight='bold',
        color=ORANGE, ha='right')
ax.text(4.066, 12, 'outside the\nsimulated data', ha='center', fontsize=11, color=GREY,
        linespacing=1.5)
ax.set_xlim(3.76, 4.12); ax.set_ylim(0, 55)
ax.set_xlabel('laminate dielectric constant (nominal)', fontsize=12.5)
ax.set_ylabel('boards passing (%)', fontsize=12.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color=FAINT, lw=1.3)
ax.set_title('The one redesign lever this data supports: the laminate', fontsize=14,
             loc='left', pad=14)
fig.tight_layout()
bad['22_laminate'] = save(fig, '22_laminate')

# ---- 13  levels, now confirmed -----------------------------------------------
fig, ax = canvas(13, 6.2)
ax.text(3, 45, '"Physics-informed" is not one thing. It has levels.', fontsize=15.5, fontweight='bold')
rows_ = [('LEVEL 1', 'Physics-blind', 'Numbers in, numbers out. No physical knowledge at all.',
          'what most people do', GREY, WHITE),
         ('LEVEL 2', 'Physics-constrained',
          'The prediction is forced to obey the laws the measurement\nobeys: reciprocity, passivity, smoothness.',
          'CONFIRMED BY HUAWEI', ORANGE, TINT_O),
         ('LEVEL 3', 'Full PINN',
          'The network predicts the fields themselves and is penalised\nfor breaking Maxwell at every point in space.',
          'not required', BLUE, TINT_B)]
for i, (n_, t_, d_, tag, col, bg) in enumerate(rows_):
    y = 30 - i*12.5
    ax.add_patch(FancyBboxPatch((3, y), 68, 10.5, boxstyle='round,pad=0.5,rounding_size=1',
                                fc=bg, ec=col, lw=2.4))
    ax.text(6, y + 7.6, n_, fontsize=10, fontweight='bold', color=col)
    ax.text(6, y + 3.2, t_, fontsize=13.5, fontweight='bold')
    ax.text(24, y + 5.2, d_, fontsize=10.8, va='center', linespacing=1.6)
    ax.text(73, y + 5.2, tag, fontsize=10.8, va='center', color=col,
            fontweight='bold' if i == 1 else 'normal')
bad['13_levels'] = save(fig, '13_levels')

# ---- 15  scoresheet with real numbers ----------------------------------------
fig, ax = canvas(13, 6.6)
ax.text(3, 48, 'What the marking scheme asks, and what answers it', fontsize=15.5, fontweight='bold')
ax.text(3, 43, 'At Huawei\'s 30 minutes per simulation, run one after another.', fontsize=11.5, color=GREY)
rows_ = [('CRITERION 1', 'Accelerated design, against\nconventional engineering',
          'Redesign search: 1.9 million designs\nin 5 minutes, not 110 years', ORANGE),
         ('CRITERION 2', 'Sensitivity analysis, against\nconventional Monte Carlo',
          '153,248 cases in 27 seconds,\nnot 8.7 years', GREEN),
         ('STATED CHALLENGE', 'How much training data\nthe model needs',
          '60 designs: 30 solver-hours\ninstead of 270', BLUE)]
for i, (a_, b_, c_, col) in enumerate(rows_):
    y = 28 - i*12.5
    ax.add_patch(FancyBboxPatch((3, y), 38, 10.5, boxstyle='round,pad=0.5,rounding_size=1',
                                fc=FAINT, ec=GREY, lw=1.6))
    ax.text(6, y + 7.8, a_, fontsize=9.5, fontweight='bold', color=GREY)
    ax.text(6, y + 3.4, b_, fontsize=11.5, va='center', linespacing=1.6)
    link(ax, 42, y + 5.2, 50, y + 5.2, col, 2.6)
    ax.add_patch(FancyBboxPatch((51, y), 45, 10.5, boxstyle='round,pad=0.5,rounding_size=1',
                                fc=WHITE, ec=col, lw=2.4))
    ax.text(54, y + 5.2, c_, fontsize=11.5, va='center', fontweight='bold', linespacing=1.6)
bad['15_criteria'] = save(fig, '15_criteria')

# ---- 16  the eleven numbers: nominal plus tolerance --------------------------
fig, ax = canvas(13, 6.4)
ax.text(3, 46, 'Each of the eleven numbers is a nominal plus a factory tolerance',
        fontsize=15.5, fontweight='bold')
ax.text(3, 41.5, 'The designer picks the nominal. The factory adds scatter within the tolerance. '
                 'Tolerances from Huawei.', fontsize=11, color=GREY)
ax.text(6, 35.5, 'PARAMETER', fontsize=9.5, fontweight='bold', color=GREY)
ax.text(40, 35.5, 'NOMINAL', fontsize=9.5, fontweight='bold', color=GREY)
ax.text(62, 35.5, 'REAL TOLERANCE', fontsize=9.5, fontweight='bold', color=GREY)
table = [('material constant', '3.91', '2%', False),
         ('board thickness', 'as drawn', '10%', True),
         ('copper thickness', '1.87', '1%', False),
         ('copper widths A1 to A9', 'not supplied', '5%', True),
         ('resistors 1, 2 and load', '1100, 342, 50 ohm', '5%', False)]
y = 30
for n_, nom_, tol_, key in table:
    ax.add_patch(FancyBboxPatch((3, y - 3.6), 93, 5.0, boxstyle='round,pad=0.2,rounding_size=0.6',
                                fc=TINT_O if key else FAINT, ec='none'))
    ax.text(6, y - 1.1, n_, fontsize=11.5, va='center', fontweight='bold')
    ax.text(40, y - 1.1, nom_, fontsize=11, va='center')
    ax.text(62, y - 1.1, tol_, fontsize=11.5, va='center', fontweight='bold',
            color=ORANGE if key else INK)
    y -= 6.4
ax.text(3, 1.5, 'Shaded rows hold the parameters that drive most of the variation.',
        fontsize=11, color=GREY)
bad['16_parameters'] = save(fig, '16_parameters')

# ---- 17  what Huawei answered -----------------------------------------------
fig, ax = canvas(13, 6.6)
ax.text(3, 48, 'What Huawei answered, and what it changed', fontsize=15.5, fontweight='bold')
rows_ = [('ONE SIMULATION COSTS', 'About 30 minutes,\nfull frequency sweep',
          'Every speed claim now\nhas a real number behind it', GREEN),
         ('DOES PORT PHYSICS COUNT?', 'Yes',
          'No separate field-based\nnetwork is needed', ORANGE),
         ('WHICH PARAMETERS VARY?', 'All eleven, each with\na real tolerance',
          'Sensitivity, yield and\nredesign were re-run', BLUE)]
for i, (a_, b_, c_, col) in enumerate(rows_):
    y = 30 - i*13.5
    ax.add_patch(FancyBboxPatch((3, y), 38, 11, boxstyle='round,pad=0.5,rounding_size=1',
                                fc=FAINT, ec=GREY, lw=1.6))
    ax.text(6, y + 8.4, a_, fontsize=9.5, fontweight='bold', color=GREY)
    ax.text(6, y + 3.6, b_, fontsize=11.5, va='center', fontweight='bold', linespacing=1.6)
    link(ax, 42, y + 5.5, 50, y + 5.5, col, 2.6)
    ax.add_patch(FancyBboxPatch((51, y), 45, 11, boxstyle='round,pad=0.5,rounding_size=1',
                                fc=WHITE, ec=col, lw=2.4))
    ax.text(54, y + 5.5, c_, fontsize=11.5, va='center', linespacing=1.6)
bad['17_answers'] = save(fig, '17_answers')

# ---- 20  what to do next -------------------------------------------------------
fig, ax = canvas(13, 5.6)
ax.text(3, 41, 'What to do next, in order', fontsize=15.5, fontweight='bold')
steps = [('NEXT', 'Simulate designs\nabove DK 4.02', 'the best laminate\nsits at the data edge', VERM),
         ('THEN', 'Confirm the laminate\nin the real solver', 'turns a prediction\ninto a result', ORANGE),
         ('ASK HUAWEI', 'Nominal copper\nwidths', 'unlocks redesign\nof the widths', BLUE),
         ('ASK HUAWEI', 'Uniform or\nbell-curve scatter?', 'the pass rate is\n32% or 70%', GREEN),
         ('LAST', 'Write it around\nthe two ratios', 'the criteria are\nthe outline', INK)]
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
    ax.text(x + 8, 9.0, why, ha='center', fontsize=9.6, color=GREY, va='center', linespacing=1.6)
bad['20_plan'] = save(fig, '20_plan')

print('\nproblems:', {k: v for k, v in bad.items() if v} or 'none')
