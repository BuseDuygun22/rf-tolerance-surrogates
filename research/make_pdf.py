"""Architecture report as a PDF. Every page is composed on one canvas and passes
the same no-overlap check as the figures: no text on text, no text on an image,
no connector through a label, nothing off the page."""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from vis import *

PW, PH = 13.333, 7.5
YMAX = 100 * PH / PW
COL = [4, 37, 70]
COL_W = 28
FULL_W = 92
pages, problems = [], {}


def page():
    return canvas(PW, PH)


def head(ax, kicker, title):
    ax.text(4, YMAX - 5.0, kicker, fontsize=9.5, fontweight='bold', color=ORANGE)
    ax.text(4, YMAX - 10.0, title, fontsize=17, fontweight='bold')
    ax.plot([4, 96], [YMAX - 12.6, YMAX - 12.6], color=FAINT, lw=2)


def body(ax, x, y, text, width=COL_W, size=10.6, color=INK, bold=False, raw=False):
    """Paragraph wrapped to `width` canvas units. 100 units span the page."""
    if raw:
        out = text
    else:
        per = (0.66 if bold else 0.60) * size / 10.6   # bold glyphs run wider
        cap = max(12, int(width / per))
        out = textwrap.fill(' '.join(text.split()), cap)
    ax.text(x, y, out, fontsize=size, va='top', linespacing=1.75, color=color,
            fontweight='bold' if bold else 'normal')


def foot(ax, y, text, color=ORANGE):
    body(ax, 4, y, text, width=FULL_W, color=color, bold=True)


def finish(fig, name):
    problems[name] = check(fig, name)
    pages.append(fig)


# ---- cover -------------------------------------------------------------------
fig, ax = page()
ax.add_patch(Rectangle((0, 0), 100, YMAX, fc=INK, ec='none', zorder=0))
ax.text(8, 38, 'HUAWEI TECH ARENA 2026    SUB-TOPIC 3', fontsize=11,
        fontweight='bold', color=ORANGE, zorder=3)
ax.text(8, 29, 'The architecture,\ncomponent by component', fontsize=26,
        fontweight='bold', color=WHITE, zorder=3, va='top', linespacing=1.4)
ax.text(8, 16, 'What each part of the system is, why it is there, and what it earns\n'
               'against the two criteria the competition actually marks.',
        fontsize=13, color='#C8D3D9', va='top', linespacing=1.8, zorder=3)
ax.text(8, 6, 'Built and measured on the 540-design dataset supplied by Huawei.',
        fontsize=10.5, color='#8E9BA3', zorder=3)
finish(fig, 'p01_cover')

# ---- the problem -------------------------------------------------------------
fig, ax = page()
head(ax, 'THE PROBLEM', 'One simulation is cheap. A hundred thousand are not.')
image(ax, 'figs/08_slow_fast.png', 4, 17, 92, 24)
body(ax, COL[0], 14, 'A full-wave simulator solves Maxwell equations for the fields '
                     'throughout the structure, then reports what comes out of each '
                     'port. It is accurate and it is slow.')
body(ax, COL[1], 14, 'That is fine when you need one answer. It is hopeless when you '
                     'need to know how a hundred thousand manufactured boards will '
                     'behave, which is what a tolerance study asks.')
body(ax, COL[2], 14, 'The trained model never solves Maxwell. It learned the pattern '
                     'from examples that did, so it answers in a tenth of a millisecond '
                     'instead of 30 minutes.', color=ORANGE, bold=True)
finish(fig, 'p02_problem')

# ---- architecture ------------------------------------------------------------
fig, ax = page()
head(ax, 'THE SYSTEM', 'How the whole thing fits together')
image(ax, 'figs/09_architecture.png', 4, 11, 92, 30)
body(ax, COL[0], 8.5, 'STEP 1 is paid once. Sixty simulations, 30 solver-hours, then '
                      'about four minutes of training. Everything after is cheap.')
body(ax, COL[1], 8.5, 'STEP 2 is where physics enters. Three laws the measurement '
                      'already obeys are built in or verified, not learned.')
body(ax, COL[2], 8.5, 'STEP 3 is the payoff. Three studies that were previously '
                      'unaffordable, each now taking seconds.', color=ORANGE, bold=True)
finish(fig, 'p03_architecture')

# ---- component 1: the data ---------------------------------------------------
fig, ax = page()
head(ax, 'COMPONENT 1', 'The training data, and how little of it you need')
image(ax, 'figs/14_dataeff.png', 3, 13, 52, 29)
body(ax, 60, 40,
     'WHAT IT IS\nDesigns simulated in the real solver,\neach an input and its measured answer.'
     '\n\nWHY IT MATTERS\nSimulations are the only real cost in\nthe project. Every one you avoid\nis time and money saved.'
     '\n\nWHAT WE MEASURED\nSixty designs reach engineering\naccuracy. That is one ninth of what\nHuawei supplied.',
     raw=True)
foot(ax, 9.5, 'The brief names data availability and training cost as two of its four technical '
              'challenges. Sixty designs is 30 solver-hours instead of 270, a claim a jury can check.')
finish(fig, 'p04_data')

# ---- component 2: the physics ------------------------------------------------
fig, ax = page()
head(ax, 'COMPONENT 2', 'The physics, and an honest negative result')
image(ax, 'figs/11_laws.png', 4, 19, 92, 22)
body(ax, COL[0], 16, 'Reciprocity is imposed by construction. The model predicts six '
                     'numbers and copies the other three, so the law can never be '
                     'broken and the problem shrinks by a third.')
body(ax, COL[1], 16, 'Passivity was tested as a training constraint, as a penalty and '
                     'as a hard projection. Neither changed the error to four decimals, '
                     'and both cost thirty times the training time.')
body(ax, COL[2], 16, 'Reporting that is worth more than hiding it. A constraint only '
                     'pays where the model actually breaks it, and this device never '
                     'came close.', color=ORANGE, bold=True)
finish(fig, 'p05_physics')

# ---- component 3: the model --------------------------------------------------
fig, ax = page()
head(ax, 'COMPONENT 3', 'Which model, and why that one')
image(ax, 'figs/19_models.png', 3, 13, 54, 29)
body(ax, 62, 40,
     'HOW THIS WAS DECIDED\nSix families, identical splits, five\nrepeats, 140 designs never seen.'
     '\n\nTHE SURPRISE\nA plain linear fit came within a\nfactor of two of the best model. Any\nproposal that skips that baseline is\nexposed.'
     '\n\nTHE CORRECTION\nCompressing the response first was\nmy own mistake. Removing it improved\nthe winner threefold.',
     raw=True)
foot(ax, 9.5, 'The recommendation rests on measurement rather than fashion. Kriging also carries a '
              'confidence estimate, which is what an active-learning loop needs and no rival here provides.')
finish(fig, 'p06_model')

# ---- output 1: sensitivity ---------------------------------------------------
fig, ax = page()
head(ax, 'OUTPUT 1', 'Which manufacturing errors actually matter')
image(ax, 'figs/10_sensitivity.png', 4, 18, 92, 24)
body(ax, COL[0], 15, 'Run under the real factory tolerances Huawei supplied: 53,248 cases '
                     'in 11 seconds. On the solver that is 3 years of simulation, run one '
                     'after another.')
body(ax, COL[1], 15, 'Board thickness and copper width A7 still explain 89 percent of '
                     'reflection. For isolation, the material constant fell from first to '
                     'second place once its real 2 percent tolerance replaced the wider '
                     'range in the dataset.')
body(ax, COL[2], 15, 'Resistors and copper thickness show no measurable effect on reflection '
                     'or coupling, even at their full real tolerance.', color=ORANGE, bold=True)
finish(fig, 'p07_sensitivity')

# ---- output 2: yield ---------------------------------------------------------
fig, ax = page()
head(ax, 'OUTPUT 2', 'How many boards pass, and how to raise it')
image(ax, 'figs/12_yield.png', 4, 17, 92, 25)
body(ax, COL[0], 14, 'A hundred thousand virtual boards under the real tolerances took 16 '
                     'seconds, against almost 6 years on the solver. Today 32 percent pass. '
                     'The model agreed with the real simulator on pass or fail for 93 '
                     'percent of designs it never saw.')
body(ax, COL[1], 14, 'Changing the laminate lifts that to 45 percent. Halving the tolerance '
                     'on two copper widths and the board thickness lifts it to 83 percent. '
                     'Each step keeps the one before it.')
body(ax, COL[2], 14, 'These are predictions. The pass rate also depends on whether factory '
                     'scatter is uniform or bell-shaped: 32 percent or 70 percent for the '
                     'same design.', color=VERM, bold=True)
finish(fig, 'p08_yield')

# ---- output 3: the laminate --------------------------------------------------
fig, ax = page()
head(ax, 'OUTPUT 3', 'The redesign: one lever, and where it runs out')
image(ax, 'figs/22_laminate.png', 4, 17, 92, 25)
body(ax, COL[0], 14, 'The search tried 1.9 million designs in 5 minutes. On the solver that '
                     'is 110 years. Only the laminate and copper thickness could move, since '
                     'the data covers every other parameter at just its own tolerance.')
body(ax, COL[1], 14, 'Raising the laminate dielectric constant from 3.91 to 4.02 cuts '
                     'isolation failures from 24 to 3 percent. Copper thickness changed the '
                     'pass rate by barely one point.')
body(ax, COL[2], 14, 'The best value sits at the edge of the simulated data. Simulating '
                     'laminates above 4.02 is the next job, and exactly what active learning '
                     'is for.', color=VERM, bold=True)
finish(fig, 'p08b_laminate')

# ---- the confirmed answer ----------------------------------------------------
fig, ax = page()
head(ax, 'CONFIRMED', 'The physics this project uses counts, and Huawei said so')
image(ax, 'figs/13_levels.png', 4, 17, 92, 24)
body(ax, COL[0], 14, 'A full physics-informed network is penalised for breaking Maxwell '
                     'equations at points in space. That needs the fields throughout the '
                     'structure, which the supplied file does not contain.')
body(ax, COL[1], 14, 'This project enforces the laws the port measurement obeys instead: '
                     'reciprocity by construction, passivity checked on every prediction.')
body(ax, COL[2], 14, 'Asked directly, Huawei confirmed that this counts as physics-informed. '
                     'No separate field-based network is needed.', color=ORANGE, bold=True)
finish(fig, 'p09_confirmed')

# ---- what Huawei answered ----------------------------------------------------
fig, ax = page()
head(ax, 'THE ANSWERS', 'What Huawei answered, and what it changed')
image(ax, 'figs/17_answers.png', 4, 11, 92, 30)
body(ax, COL[0], 8.5, 'The 30-minute cost turns every speed claim from an expression into a '
                      'number a jury can check.')
body(ax, COL[1], 8.5, 'The yes on physics removes the biggest risk this project carried.')
body(ax, COL[2], 8.5, 'The tolerance table changed two conclusions, and both are reported.',
     color=ORANGE, bold=True)
finish(fig, 'p10_answers')

# ---- the scoresheet ----------------------------------------------------------
fig, ax = page()
head(ax, 'THE SCORESHEET', 'What the marking scheme rewards, and what answers it')
image(ax, 'figs/15_criteria.png', 4, 15, 92, 26)
body(ax, COL[0], 12, 'Both criteria compare your cost with the conventional cost. At 30 '
                     'minutes a simulation, both ratios are now real numbers.')
body(ax, COL[1], 12, 'All times are serial. Even with 100 simulations running at once, the '
                     'sensitivity study alone would take 11 days.')
body(ax, COL[2], 12, 'Quote time units, not a multiplier. Years against minutes reads as '
                     'credible where a factor of millions does not.', color=ORANGE, bold=True)
finish(fig, 'p11_scoresheet')

# ---- plan --------------------------------------------------------------------
fig, ax = page()
head(ax, 'THE PLAN', 'What to do next, in order')
image(ax, 'figs/20_plan.png', 4, 17, 92, 24)
body(ax, 4, 14, 'The first two steps turn the redesign from a prediction into a result. '
                'The two questions for Huawei would lift the last limits on what the '
                'data can say.', width=44)
body(ax, 52, 14, 'The last step is not writing up what you did. It is writing the '
                 'submission around the two ratios the jury scores, and letting the '
                 'evidence fall under whichever ratio it supports.', width=44,
     color=ORANGE, bold=True)
finish(fig, 'p13_plan')

with PdfPages('Huawei_TechArena_Architecture.pdf') as pdf:
    for f in pages:
        pdf.savefig(f)
        plt.close(f)

bad = {k: v for k, v in problems.items() if v}
print('\npages:', len(pages))
print('page problems:', bad if bad else 'none')
print('saved Huawei_TechArena_Architecture.pdf')
