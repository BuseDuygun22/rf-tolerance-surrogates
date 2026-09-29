import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from vis import *
bad = {}

# ---- 13  levels of physics-informed -----------------------------------------
fig, ax = canvas(13, 6.2)
ax.text(3, 45, '"Physics-informed" is not one thing. It has levels.',
        fontsize=15.5, fontweight='bold')
rows = [('LEVEL 1', 'Physics-blind',
         'Numbers in, numbers out. No physical knowledge at all.',
         'what most people do', GREY, WHITE),
        ('LEVEL 2', 'Physics-constrained',
         'The prediction is forced to obey the laws the measurement\nobeys: reciprocity, passivity, smoothness.',
         'WHAT YOUR DATA SUPPORTS', ORANGE, TINT_O),
        ('LEVEL 3', 'Full PINN',
         'The network predicts the fields themselves and is penalised\nfor breaking Maxwell at every point in space.',
         'needs field data you do not have', BLUE, TINT_B)]
for i, (n_, t_, d_, tag, col, bg) in enumerate(rows):
    y = 30 - i*12.5
    ax.add_patch(FancyBboxPatch((3, y), 68, 10.5,
                                boxstyle='round,pad=0.5,rounding_size=1', fc=bg, ec=col, lw=2.4))
    ax.text(6, y + 7.6, n_, fontsize=10, fontweight='bold', color=col)
    ax.text(6, y + 3.2, t_, fontsize=13.5, fontweight='bold')
    ax.text(24, y + 5.2, d_, fontsize=10.8, va='center', linespacing=1.6)
    ax.text(73, y + 5.2, tag, fontsize=10.8, va='center', color=col,
            fontweight='bold' if i == 1 else 'normal')
bad['13_levels'] = save(fig, '13_levels')

# ---- 15  what the marking scheme rewards ------------------------------------
fig, ax = canvas(13, 6.6)
ax.text(3, 48, 'What the marking scheme asks, and what answers it',
        fontsize=15.5, fontweight='bold')
ax.text(3, 43, 'Both criteria are ratios. You hold one half. Huawei holds the other.',
        fontsize=11.5, color=VERM)
rows = [('CRITERION 1', 'Accelerated design, against\nconventional engineering',
         'Redesign for yield:\n8 million candidates in 150 s', ORANGE),
        ('CRITERION 2', 'Sensitivity analysis, against\nconventional Monte Carlo',
         '153,000 cases in 18.6 s,\nagainst 106 days of solver', GREEN),
        ('STATED CHALLENGE', 'How much training data\nthe model needs',
         '60 of the 540 designs,\nabout one ninth', BLUE)]
for i, (a_, b_, c_, col) in enumerate(rows):
    y = 28 - i*12.5
    ax.add_patch(FancyBboxPatch((3, y), 38, 10.5,
                                boxstyle='round,pad=0.5,rounding_size=1', fc=FAINT, ec=GREY, lw=1.6))
    ax.text(6, y + 7.8, a_, fontsize=9.5, fontweight='bold', color=GREY)
    ax.text(6, y + 3.4, b_, fontsize=11.5, va='center', linespacing=1.6)
    link(ax, 42, y + 5.2, 50, y + 5.2, col, 2.6)
    ax.add_patch(FancyBboxPatch((51, y), 45, 10.5,
                                boxstyle='round,pad=0.5,rounding_size=1', fc=WHITE, ec=col, lw=2.4))
    ax.text(54, y + 5.2, c_, fontsize=11.5, va='center', fontweight='bold', linespacing=1.6)
bad['15_criteria'] = save(fig, '15_criteria')

# ---- 16  the eleven parameters ----------------------------------------------
fig, ax = canvas(13, 7.2)
ax.text(3, 53, 'The eleven numbers that describe one board', fontsize=15.5, fontweight='bold')
y = 46
for title, col, tint, items in [
        ('THE FACTORY CANNOT CONTROL THESE', GREY, FAINT,
         [('material constant', '3.70 to 4.10', 'how much the board slows the wave'),
          ('board thickness', 'plus or minus 10%', 'the insulating layer'),
          ('copper thickness', '1.40 to 2.40', 'the plating')]),
        ('THE DESIGNER CHOOSES THESE', ORANGE, TINT_O,
         [('copper widths A1 to A9', 'plus or minus 1', 'five separate track segments'),
          ('three resistors', '1100, 342, 50 ohm', 'soldered components')])]:
    ax.text(3, y, title, fontsize=10.5, fontweight='bold', color=col)
    y -= 6.0
    for n_, r_, d_ in items:
        ax.add_patch(FancyBboxPatch((3, y - 3.6), 93, 5.0,
                                    boxstyle='round,pad=0.2,rounding_size=0.6', fc=tint, ec='none'))
        ax.text(6, y - 1.1, n_, fontsize=11.5, va='center', fontweight='bold')
        ax.text(34, y - 1.1, r_, fontsize=11, va='center', color=col)
        ax.text(57, y - 1.1, d_, fontsize=11, va='center', color=GREY)
        y -= 6.2
    y -= 2.0
ax.text(3, 2.0, 'All eleven are drawn at random and independently, to imitate real production scatter.',
        fontsize=11, color=GREY)
bad['16_parameters'] = save(fig, '16_parameters')

# ---- 17  what to do if there is no field data -------------------------------
fig, ax = canvas(13, 7.8)
ax.text(3, 57, 'If Huawei has no field data, which way do you go?',
        fontsize=15.5, fontweight='bold')
card(ax, 33, 47, 34, 8, 'ASK HUAWEI FIRST', 'is there field or geometry data?',
     fc=FAINT, ec=INK, title_size=12, sub_size=9.5)
link(ax, 42, 47, 26, 39, GREEN, 2.4)
link(ax, 58, 47, 74, 39, ORANGE, 2.4)
ax.text(23, 44.0, 'YES', fontsize=11.5, fontweight='bold', color=GREEN)
ax.text(75, 44.0, 'NO', fontsize=11.5, fontweight='bold', color=ORANGE)
card(ax, 4, 30, 42, 9, 'ADD A REAL PINN', 'train on their fields', ec=GREEN,
     title_size=12.5, sub_size=10)
card(ax, 54, 30, 42, 9, 'GENERATE YOUR OWN FIELDS', 'free solver, simpler structure', ec=ORANGE,
     title_size=12.5, sub_size=10)
for x, col, lines in [
        (4, GREEN, ['COST     a few days of extra work',
                    'GAIN     matches the brief word for word',
                    'RISK     their data may not fit the method']),
        (54, ORANGE, ['COST     one to two weeks, and a second problem',
                      'GAIN     a genuine Maxwell demonstration you own',
                      'RISK     the jury may call it a side project'])]:
    for j, ln in enumerate(lines):
        ax.text(x + 1, 25.0 - j*4.6, ln, fontsize=10.8, color=col if j == 1 else INK)
card(ax, 20, 2.5, 60, 8, 'EITHER WAY, THE TOLERANCE WORK STAYS THE CORE',
     'it is what the supplied data actually supports', ec=BLUE, lw=3,
     title_size=12, sub_size=10)
bad['17_decision'] = save(fig, '17_decision')

print('\nproblems:', {k: v for k, v in bad.items() if v})
