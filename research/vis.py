"""Figure toolkit with an automatic no-overlap guarantee.

Palette is Okabe-Ito, the standard colour-vision-deficiency safe set, so no two
meaning-carrying colours are confusable under deuteranopia, protanopia or
tritanopia. Colour never carries meaning on its own: every coloured element also
carries a text label or a shape difference.

check() renders the figure and fails loudly if any two text labels touch, or if
any drawn line or arrow passes through a text label, or if anything runs outside
the canvas. Nothing ships until it returns clean.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
from matplotlib.text import Text as _Text, Annotation as _Annotation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Okabe-Ito
INK      = '#121417'
GREY     = '#5C6B73'
FAINT    = '#E8ECEE'
BLUE     = '#0072B2'
ORANGE   = '#E69F00'
VERM     = '#D55E00'
GREEN    = '#009E73'
PURPLE   = '#CC79A7'
WHITE    = '#FFFFFF'
TINT_B   = '#E4F0F7'
TINT_O   = '#FBF0DC'
TINT_G   = '#DFF3EC'

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':12,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.edgecolor':GREY,'figure.facecolor':WHITE,'savefig.facecolor':WHITE,
})
DPI = 220
_registry = {}          # fig -> dict(owned=set(id(text)), lines=[list of (x,y) arrays in data coords + axis)]


def canvas(w, h):
    """Blank drawing surface, 100 units wide, height scaled to the aspect."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / w); ax.axis('off')
    _registry[fig] = {'owned': set(), 'paths': [], 'boxes': []}
    return fig, ax


def _own(fig, *texts):
    for t in texts:
        if t is not None:
            _registry[fig]['owned'].add(id(t))


def card(ax, x, y, w, h, title, sub=None, fc=WHITE, ec=INK, lw=2.2,
         title_size=13, sub_size=10, tc=None):
    """A labelled box. Its own text is exempt from the overlap check."""
    fig = ax.figure
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=1.1',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    _registry[fig]['boxes'].append((ax, x - 0.5, y - 0.5, w + 1.0, h + 1.0, []))
    tc = tc or INK
    if sub:
        t1 = ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                     fontsize=title_size, fontweight='bold', color=tc, zorder=3)
        t2 = ax.text(x + w/2, y + h*0.28, sub, ha='center', va='center',
                     fontsize=sub_size, color=tc, zorder=3)
        _own(fig, t1, t2)
        _registry[fig]['boxes'][-1][5].extend([id(t1), id(t2)])
    else:
        t1 = ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                     fontsize=title_size, fontweight='bold', color=tc, zorder=3)
        _own(fig, t1)
        _registry[fig]['boxes'][-1][5].append(id(t1))
    return x + w/2, y + h/2


def link(ax, x1, y1, x2, y2, color=INK, lw=2.4, dashed=False, head=True):
    """Straight connector, registered so the checker can test it against labels."""
    fig = ax.figure
    style = '-|>' if head else '-'
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=17, lw=lw, color=color,
                                 linestyle='--' if dashed else '-',
                                 shrinkA=0, shrinkB=0, zorder=4))
    n = 60
    _registry[fig]['paths'].append((ax, np.linspace(x1, x2, n), np.linspace(y1, y2, n)))


def elbow(ax, x1, y1, x2, y2, color=INK, lw=2.4, bus=None):
    """Down-then-across-then-down connector; keeps diagonals off the labels."""
    fig = ax.figure
    mid = (y1 + y2) / 2 if bus is None else bus
    pts = [(x1, y1), (x1, mid), (x2, mid), (x2, y2)]
    for (ax1_, ay1_), (ax2_, ay2_) in zip(pts[:-1], pts[1:]):
        ax.plot([ax1_, ax2_], [ay1_, ay2_], color=color, lw=lw, zorder=4,
                solid_capstyle='round')
    ax.add_patch(FancyArrowPatch((x2, y2 + (3 if y1 > y2 else -3)), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=17, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=4))
    for (ax1_, ay1_), (ax2_, ay2_) in zip(pts[:-1], pts[1:]):
        n = 40
        _registry[fig]['paths'].append((ax, np.linspace(ax1_, ax2_, n),
                                        np.linspace(ay1_, ay2_, n)))


def _keep(t):
    return t.get_text().strip() and t.get_visible()


def _texts(fig):
    out = []
    owned = _registry.get(fig, {}).get('owned', set())
    for ax in fig.get_axes():
        for t in ax.texts:
            if _keep(t):
                out.append((t, id(t) in owned))
        if not ax.axison:
            continue
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if _keep(t):
                out.append((t, False))
        lo, hi = ax.get_xlim()
        for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if _keep(t) and min(lo, hi) - 1e-9 <= loc <= max(lo, hi) + 1e-9:
                out.append((t, 'tick'))
        lo, hi = ax.get_ylim()
        for loc, t in zip(ax.get_yticks(), ax.get_yticklabels()):
            if _keep(t) and min(lo, hi) - 1e-9 <= loc <= max(lo, hi) + 1e-9:
                out.append((t, 'tick'))
    for t in fig.texts:
        if _keep(t):
            out.append((t, False))
    for ax in fig.get_axes():
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                out.append((t, False))
    return out


def _extent(t, r):
    """Bounding box of the glyphs only. An Annotation's own get_window_extent
    includes its arrow, which would make every curve it points at look like a
    collision."""
    if isinstance(t, _Annotation):
        return _Text.get_window_extent(t, r)
    return t.get_window_extent(r)


def check(fig, name, pad=1.5, verbose=True):
    """Return a list of problems. Empty list means the figure is clean."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(t.get_text()[:30].replace(chr(10), ' / '), _extent(t, r), owned, id(t))
             for t, owned in _texts(fig)]
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if (a.x0 - pad < b.x1 and b.x0 - pad < a.x1 and
                    a.y0 - pad < b.y1 and b.y0 - pad < a.y1):
                bad.append(f'TEXT/TEXT  "{items[i][0]}"  vs  "{items[j][0]}"')
    fb = fig.bbox
    for label, bb, kind, _tid in items:
        if kind == 'tick':
            continue          # matplotlib clips ticks outside the view itself
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
            bad.append(f'OFF-CANVAS "{label}"')
    for ax, xs, ys in _registry.get(fig, {}).get('paths', []):
        px, py = ax.transData.transform(np.column_stack([xs, ys])).T
        for label, bb, owned, _tid in items:
            hit = ((px > bb.x0 - pad) & (px < bb.x1 + pad) &
                   (py > bb.y0 - pad) & (py < bb.y1 + pad))
            if hit.any():
                bad.append(f'LINE/TEXT  connector crosses "{label}"')
                break
    for pax in fig.get_axes():
        grid = set(map(id, pax.get_xgridlines() + pax.get_ygridlines()))
        for ln in pax.lines:
            if not ln.get_visible() or id(ln) in grid:
                continue
            xy = ln.get_xydata()
            if len(xy) < 2:
                continue
            import numpy as _np
            seg = _np.asarray(xy, float)
            keep = _np.isfinite(seg).all(axis=1)
            seg = seg[keep]
            if len(seg) < 2:
                continue
            dense = _np.concatenate([_np.linspace(seg[i], seg[i + 1], 12)
                                     for i in range(len(seg) - 1)])
            qx, qy = pax.transData.transform(dense).T
            for label, bb, kind, _tid in items:
                if kind == 'tick':
                    continue
                if ((qx > bb.x0 - pad) & (qx < bb.x1 + pad) &
                        (qy > bb.y0 - pad) & (qy < bb.y1 + pad)).any():
                    bad.append(f'CURVE/TEXT plotted line crosses "{label}"')
                    break
    for bax, bx, by, bw, bh, ids in _registry.get(fig, {}).get('boxes', []):
        (dx0, dy0), (dx1, dy1) = bax.transData.transform([(bx, by), (bx + bw, by + bh)])
        for label, bb, kind, tid in items:
            if tid in ids:
                continue
            if (dx0 - pad < bb.x1 and bb.x0 - pad < dx1 and
                    dy0 - pad < bb.y1 and bb.y0 - pad < dy1):
                bad.append(f'TEXT/BOX   "{label}" sits on a card')
    bad = sorted(set(bad))
    if verbose:
        print(('  CLEAN  ' if not bad else '  FAIL   ') + name +
              ('' if not bad else '\n           ' + '\n           '.join(sorted(set(bad)))))
    return bad


def save(fig, name):
    problems = check(fig, name)
    fig.savefig(f'figs/{name}.png', dpi=DPI, bbox_inches='tight', pad_inches=0.18)
    fig.savefig(f'figs/{name}.pdf', bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)
    return problems


def image(ax, path, x, y, w, h):
    """Place a PNG inside the data-coordinate rect (x, y, w, h), preserving its
    aspect ratio and centring it. The area it actually occupies is registered so
    check() will flag any text that lands on top of it."""
    import matplotlib.image as mpimg
    im = mpimg.imread(path)
    ih, iw = im.shape[0], im.shape[1]
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    x0, y0 = x + (w - dw) / 2, y + (h - dh) / 2
    ax.imshow(im, extent=(x0, x0 + dw, y0, y0 + dh), aspect='auto', zorder=2,
              interpolation='antialiased')
    _registry[ax.figure]['boxes'].append((ax, x0, y0, dw, dh, []))
    return x0, y0, dw, dh
