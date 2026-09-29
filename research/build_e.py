import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from vis import *
bad = {}
fig, ax = canvas(13, 7.4)
ax.text(3, 54, 'Six questions to send Huawei', fontsize=15.5, fontweight='bold')
ax.text(3, 49.5, 'The first one is worth more than the other five together.',
        fontsize=11.5, color=VERM)
qs = [('What does one full-wave simulation of this component cost?',
       'Every speed claim in the brief is a ratio against this number.', VERM),
      ('What is the acceptance specification across 1.2 to 1.7 GHz?',
       'Reflection, coupling and isolation limits. Yield is meaningless without it.', INK),
      ('Which parameters are design choices, and which are process noise?',
       'This split decides the whole redesign study.', INK),
      ('What is the nominal design the tolerances perturb?',
       'The widths are deviations from a value the file does not contain.', INK),
      ('Do you have field or geometry data for these structures?',
       'If yes, a true physics-informed network becomes possible.', BLUE),
      ('What are the units of the copper thickness and the width deviations?',
       'Needed before any tolerance advice can reach a manufacturing engineer.', INK)]
y = 43
for i, (q, why, col) in enumerate(qs):
    ax.add_patch(Circle((5.5, y - 1.6), 2.0, fc=col, ec='none', zorder=2))
    ax.text(5.5, y - 1.6, str(i + 1), ha='center', va='center', color=WHITE,
            fontsize=11, fontweight='bold', zorder=3)
    ax.text(10, y, q, fontsize=12.2, fontweight='bold', va='top')
    ax.text(10, y - 3.4, why, fontsize=10.8, color=GREY, va='top')
    y -= 7.2
bad['18_questions'] = save(fig, '18_questions')
print('problems:', {k: v for k, v in bad.items() if v})
