"""End-to-end pipeline v2, with the evidence behind each stage."""
from vis import *

bad = {}
fig, ax = canvas(13.5, 8.2)
ax.text(2, 58.5, 'The pipeline, end to end', fontsize=16, fontweight='bold')
ax.text(2, 54.8, 'Each stage carries the number that justifies it, measured on the 540 supplied designs.',
        fontsize=11, color=GREY)

W, H = 30.5, 12.5
row1, row2 = 38.5, 18.0
xs_ = [2, 34.75, 67.5]
stages1 = [('0   DATA AUDIT', 'reciprocity holds to 1e-11, passive\neverywhere, smooth in frequency', INK),
           ('1   PHYSICS-CONSISTENT MODEL', 'kriging, 6 of 9 outputs by reciprocity,\npassivity checked on every prediction', BLUE),
           ('2   TRUST GATE', '93% pass/fail agreement on unseen\nboards, holds beyond the laminate range', BLUE)]
stages2 = [('3   SENSITIVITY, NO MONTE CARLO', 'closed-form indices in 0.3 s, 3 to 9x\ncloser than Monte Carlo per run', ORANGE),
           ('4   PASS RATE', '40 simulations match about 800\ndirect Monte Carlo runs', ORANGE),
           ('5   DESIGN FOR YIELD', 'laminate, then the cheapest tolerances:\n31% to 83% pass under stated limits', GREEN)]
for x, (t_, s_, c) in zip(xs_, stages1):
    card(ax, x, row1, W, H, t_, s_, ec=c, lw=2.4, title_size=11.5, sub_size=9.6)
for x, (t_, s_, c) in zip(xs_, stages2):
    card(ax, x, row2, W, H, t_, s_, ec=c, lw=2.4, title_size=11.5, sub_size=9.6)
for i in range(2):
    link(ax, xs_[i] + W + 0.6, row1 + H / 2, xs_[i + 1] - 0.6, row1 + H / 2, INK, 2.2)
    link(ax, xs_[i] + W + 0.6, row2 + H / 2, xs_[i + 1] - 0.6, row2 + H / 2, INK, 2.2)
elbow(ax, xs_[2] + W / 2, row1 - 0.6, xs_[0] + W / 2, row2 + H + 0.6, INK, 2.2, bus=row1 - 3.2)

card(ax, 2, 1.5, 96, 9.5, '6   CHECK AGAINST THE RAW SIMULATIONS',
     'real boards from the 540 that fall inside each scenario reproduce all five predicted pass rates, with no model involved',
     ec=VERM, lw=2.4, title_size=11.5, sub_size=9.8)
elbow(ax, xs_[2] + W / 2, row2 - 0.6, 50, 11.6, VERM, 2.2, bus=row2 - 2.6)
bad['23_pipeline_v2'] = save(fig, '23_pipeline_v2')
print('problems:', {k: v for k, v in bad.items() if v} or 'none')
