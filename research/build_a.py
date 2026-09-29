"""Concept diagrams. Every figure is verified for zero overlap before it is written."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from vis import *

bad_total = {}

# ---- 01  the component ------------------------------------------------------
fig, ax = canvas(12, 5.6)
ax.text(3, 43, 'The object your data describes', fontsize=16, fontweight='bold')
ax.text(3, 38.5, 'A three-port coupler. It taps a small sample of a signal without interrupting it.',
        fontsize=11.5, color=GREY)
card(ax, 37, 14, 26, 14, 'COUPLER', fc=TINT_B, ec=BLUE, title_size=15)
link(ax, 5, 21, 36, 21, BLUE, 3)
ax.text(5, 25.5, 'PORT 1', fontsize=12, fontweight='bold', color=BLUE)
ax.text(5, 16.5, 'signal enters', fontsize=11, color=GREY)
link(ax, 64, 21, 95, 21, BLUE, 3)
ax.text(70, 25.5, 'PORT 2', fontsize=12, fontweight='bold', color=BLUE)
ax.text(70, 16.5, '57% carries on', fontsize=11, color=GREY)
link(ax, 50, 13, 50, 5, ORANGE, 3)
ax.text(54, 9.5, 'PORT 3', fontsize=12, fontweight='bold', color=ORANGE)
ax.text(54, 5.5, '5% is tapped off and measured', fontsize=11, color=GREY)
bad_total['01_component'] = save(fig, '01_component')

# ---- 02  a wave has two properties -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
t = np.linspace(0, 4*np.pi, 700)
a = axes[0]
a.plot(t, np.sin(t), color=BLUE, lw=3)
a.plot(t, 2*np.sin(t), color=ORANGE, lw=3, ls='--')
a.text(4*np.pi, 2.05, 'large', color=ORANGE, fontsize=12, fontweight='bold', ha='right', va='bottom')
a.text(4*np.pi, 1.05, 'small', color=BLUE, fontsize=12, fontweight='bold', ha='right', va='bottom')
a.set_title('AMPLITUDE   how big the wave is', fontsize=12.5, fontweight='bold', pad=14, loc='left')
b = axes[1]
b.plot(t, np.sin(t), color=BLUE, lw=3)
b.plot(t, np.sin(t-1.5), color=ORANGE, lw=3, ls='--')
b.annotate('', xy=(np.pi/2, 1.45), xytext=(np.pi/2+1.5, 1.45),
           arrowprops=dict(arrowstyle='<->', color=INK, lw=2))
b.text(np.pi/2+0.75, 1.62, 'shifted', ha='center', fontsize=12, fontweight='bold')
b.set_title('PHASE   where it is in its cycle', fontsize=12.5, fontweight='bold', pad=14, loc='left')
for a_ in axes:
    a_.set_ylim(-2.6, 2.6); a_.set_xlim(-0.2, 4*np.pi+0.2)
    a_.set_xticks([]); a_.set_yticks([])
    a_.spines[:].set_visible(False); a_.axhline(0, color=FAINT, lw=2)
fig.tight_layout()
bad_total['02_wave'] = save(fig, '02_wave')

# ---- 03  why two columns per measurement ------------------------------------
fig = plt.figure(figsize=(12, 4.8))
ax = fig.add_axes([0.04, 0.10, 0.36, 0.80])
ax.axhline(0, color=GREY, lw=1.5); ax.axvline(0, color=GREY, lw=1.5)
A, B = 1.5, 1.05
ax.add_patch(FancyArrowPatch((0, 0), (A, B), arrowstyle='-|>', mutation_scale=22, lw=3.4, color=ORANGE))
ax.plot([A, A], [0, B], ls=':', color=GREY, lw=1.8)
ax.plot([0, A], [B, B], ls=':', color=GREY, lw=1.8)
ax.text(A/2, -0.30, 'real part', ha='center', fontsize=12, color=BLUE)
ax.text(-0.10, B/2, 'imaginary\npart', ha='right', va='center', fontsize=12, color=BLUE)
ax.text(0.30, 1.32, 'length = amplitude', fontsize=12, color=ORANGE, fontweight='bold')
ax.text(0.16, 0.56, 'angle = phase', fontsize=11.5, color=INK)
ax.set_xlim(-0.75, 2.0); ax.set_ylim(-0.62, 1.62)
ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
ax2 = fig.add_axes([0.46, 0.10, 0.52, 0.80]); ax2.axis('off')
ax2.text(0, 1.0, 'Why your file has two columns per measurement',
         fontsize=15, fontweight='bold', va='top')
body = ('One number cannot carry both amplitude and phase,\n'
        'so engineers store the pair as a point on a plane.\n\n'
        'The horizontal coordinate is called the real part.\n'
        'The vertical one is called the imaginary part.\n'
        'Nothing about it is imaginary. The name is historical.')
ax2.text(0, 0.80, body, fontsize=12, va='top', linespacing=1.75)
ax2.text(0, 0.19, 'S1,1_real  and  S1,1_imag  together say how much of the\n'
                  'wave came back, and how far it slipped in its cycle.',
         fontsize=12, va='top', color=VERM, fontweight='bold', linespacing=1.75)
bad_total['03_complex'] = save(fig, '03_complex')

# ---- 05  decibels ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.4))
db = np.array([0, -3, -10, -20, -30])
pw = 10**(db/10)*100
ax.barh(range(len(db)), pw, color=[BLUE, BLUE, ORANGE, ORANGE, VERM], height=0.55)
for i, p in enumerate(pw):
    ax.text(p + 2.0, i, f'{p:g}% of the power survives', va='center', fontsize=12)
ax.set_yticks(range(len(db)))
ax.set_yticklabels([f'{d} dB' for d in db], fontsize=13, fontweight='bold')
ax.invert_yaxis(); ax.set_xlim(0, 152); ax.set_xticks([])
ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.set_title('A decibel is just a compact way of writing "what fraction got through"',
             fontsize=13.5, pad=16, loc='left')
fig.tight_layout()
bad_total['05_decibels'] = save(fig, '05_decibels')

for k, v in bad_total.items():
    pass
print('\nfigures with problems:', [k for k, v in bad_total.items() if v])
