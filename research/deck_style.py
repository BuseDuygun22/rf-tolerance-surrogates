"""Figure styling for the submission deck: transparent PNGs with a rounded white
card, colours taken from the template's own brand (charcoal logo type, red to
orange accent) plus one blue for contrast. Colour never carries meaning alone:
every mark is also labelled, and series differ in marker or dash.

Every figure still goes through vis.check() before it is written.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import vis
from vis import check, _registry

CHAR = '#3A3D40'      # charcoal, the logo type
MUTED = '#6B7378'
RULE = '#D9DEE2'
CARD = '#FFFFFF'
BRAND = '#D9480F'     # deep red-orange, readable on white
BRAND_L = '#F59A2E'   # orange, the gradient's far end
BLUE = '#0B6EA8'
GREEN = '#0A8F6A'
SOFT = '#F3F5F7'
DPI = 300

plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'text.color': CHAR,
                     'axes.labelcolor': CHAR, 'xtick.color': CHAR, 'ytick.color': CHAR,
                     'axes.edgecolor': MUTED, 'font.size': 12})


def card_fig(w, h):
    """Figure of exactly w x h inches, transparent outside a rounded white card."""
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_alpha(0)
    bg = FancyBboxPatch((0.006, 0.006), 0.988, 0.988, boxstyle='round,pad=0,rounding_size=0.028',
                        transform=fig.transFigure, fc=CARD, ec=RULE, lw=1.2, zorder=-10)
    fig.add_artist(bg)
    _registry[fig] = {'owned': set(), 'paths': [], 'boxes': []}
    return fig


def axes_in(fig, l, b, w, h, **kw):
    """Axes placed in inch-fractions of the card, transparent so the card shows."""
    ax = fig.add_axes([l, b, w, h], **kw)
    ax.patch.set_alpha(0)
    return ax


def canvas_ax(fig):
    """Free-drawing layer covering the whole card: 100 x (100*h/w) units."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / w); ax.axis('off')
    return ax


def save_card(fig, name, verbose=True):
    problems = check(fig, name, verbose=verbose)
    fig.savefig(f'figs/deck_{name}.png', dpi=DPI, transparent=True)
    plt.close(fig)
    return problems
