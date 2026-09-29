"""Submission deck built inside the TechArena 2026 template.

Structure follows the mentor's pitch guidance (problem -> solution -> how it
works -> why better, twice for the two KPIs -> validation) plus the one extra
slide Huawei's e-mail of 2026-09-22 explicitly allows for state of the art
and references, on top of the template's fixed title / QR / thank-you slides.
"""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE

SRC = '../TECHARENA 2026 - Project Template.pptx'
OUT = 'Huawei_TechArena_Submission.pptx'
CHAR = RGBColor(0x3A, 0x3D, 0x40)
MUTED = RGBColor(0x6B, 0x73, 0x78)
BRAND = RGBColor(0xD9, 0x48, 0x0F)

prs = Presentation(SRC)
S = list(prs.slides)                                    # the 7 template slides, original order
LAYOUTS = {l.name: l for l in prs.slide_masters[0].slide_layouts}


def ph(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def drop(shape):
    shape._element.getparent().remove(shape._element)


def set_text(shape, paras, size, bold=False, color=None, align=None, space_after=6, bullets=None):
    tf = shape.text_frame
    tf.word_wrap = True
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        if align is not None:
            p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.size = Pt(size)
        r.font.bold = bold
        if color is not None:
            r.font.color.rgb = color


def add_para(shape, text, size, bold=False, color=None, space_before=14):
    """Appends one more paragraph to a shape's existing text frame (no clearing)."""
    tf = shape.text_frame
    p = tf.add_paragraph()
    p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    if color is not None:
        r.font.color.rgb = color
    return p


def title(slide, name, text, size):
    t = ph(slide, name)
    t.text_frame.clear()
    r = t.text_frame.paragraphs[0].add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    return t


def picture(slide, path, l, t, w, h, alt):
    pic = slide.shapes.add_picture(path, l, t, w, h)
    pic._element.nvPicPr.cNvPr.set('descr', alt)
    return pic


def takeaway(slide, text, top=6.34, size=15):
    tb = slide.shapes.add_textbox(Inches(0.92), Inches(top), Inches(11.5), Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = BRAND
    return tb


def footer(slide):
    for s in slide.shapes:
        if s.is_placeholder and s.placeholder_format.type is not None and 'FOOTER' in str(s.placeholder_format.type):
            s.text_frame.paragraphs[0].runs[0].text = 'HUAWEI TECHARENA 2026'
            for r in s.text_frame.paragraphs[0].runs[1:]:
                r.text = ''


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ==================================================================== slide 1  — title
s = S[0]
title(s, 'Title 1', 'Physics-checked AI for RF tolerance analysis', 32)
sub = s.shapes.add_textbox(Inches(1.2), Inches(5.72), Inches(10.93), Inches(0.5))
sub.text_frame.word_wrap = True
p = sub.text_frame.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = 'Sub-topic 3  |  A 30-minute solver run replaced by a checked surrogate model'
r.font.size = Pt(17)
r.font.color.rgb = MUTED
notes(s, 'Opening, 30 seconds. Competence: this team treated the S-parameter physics as a hard constraint on the model, not a hope. '
         'Relevance: one full-wave run takes 30 minutes, and the two things this challenge scores – sensitivity analysis and design search – '
         'both need thousands to millions of runs. Promise: today we show one model, trained on 60 runs, that stands in for both, '
         'checked against the physics and against boards it never saw.')

# ==================================================================== slide 2  — problem (NEW)
s_problem = prs.slides.add_slide(LAYOUTS['Title and Content'])
title(s_problem, 'Title 1', 'One tolerance study costs months of solver time', 30)
drop(ph(s_problem, 'Content Placeholder 2'))
picture(s_problem, 'figs/deck_problem.png', Inches(0.92), Inches(2.0), Inches(11.5), Inches(3.3),
        'Four figures: one full-wave simulation takes 30 minutes; a sensitivity analysis on the solver needs 13,312 runs, 277 days; '
        'a design search on the solver needs 1.92 million runs, 110 years; the design margin today is 0.25 to 0.40 dB against 1.3 dB of factory spread.')
takeaway(s_problem, 'Only 3 in 10 boards pass today, and finding out why with the solver alone is not affordable.', top=5.75, size=17)
notes(s_problem, 'Set up the stakes before showing the solution, about one minute. Every number here comes from the mentor-supplied solver '
      'cost (30 minutes per run) and from the dataset itself – nothing is a claim about our method yet. The margin tile is why it matters: '
      'the nominal design passes, but factory scatter is three to five times larger than its margin, so about 70% of boards fail the illustrative limits.')

# ==================================================================== slide 3  — solution (was slide 2)
s = S[1]
title(s, 'Title 1', 'A 60-run surrogate stands in for the solver', 30)
body = ph(s, 'Content Placeholder 2')
body.text_frame.clear()
set_text(body, ['Kriging model of the 11 tolerance parameters.',
                '60 solver runs are enough: ten random 60-run models reproduce the 540-run results within about 3 points.',
                'Reciprocity built in, passivity checked on every predicted matrix – confirmed by Huawei as physics-informed.',
                'One model answers both criteria: sensitivity (KPI 2) and design search (KPI 1).'],
         16, space_after=12)
drop(ph(s, 'Picture Placeholder 3'))
picture(s, 'figs/deck_pipeline.png', Inches(6.75), Inches(2.0), Inches(5.67), Inches(4.76),
        'Six-stage pipeline: audit the data, kriging with consistency checks, validation on unseen boards, '
        'exact sensitivity analysis (criterion 2), virtual design and tolerance study (criterion 1), scenario check with boards held out.')
notes(s, 'Walk the pipeline top to bottom. Be precise if asked: the validation numbers on slide 6 come from models trained on the '
         '540 boards minus the held-out ones; the 60-run claim rests on a separate test in which ten random 60-run models gave the same '
         'pass-rate staircase within 1.7 to 3.3 points and the same top two tolerances every time. No extra solver runs were available, '
         'so for each scenario we removed its real boards, retrained, and compared – that is the last stage. Polynomial chaos and a '
         'neural network agree within 2 points, which shows robustness to model choice, not correctness by itself: they share the same boards.')

# ==================================================================== slide 4  — why better: KPI 2, sensitivity (was slide 3)
s = S[2]
title(s, 'Title 1', 'KPI 2 – sensitivity: the right drivers from 30 runs, not 13,000', 28)
drop(ph(s, 'Content Placeholder 2'))
drop(ph(s, 'Content Placeholder 3'))
picture(s, 'figs/deck_h2h.png', Inches(0.92), Inches(2.0), Inches(5.67), Inches(4.25),
        'Line chart of error in sensitivity indices against real simulations used. Conventional Monte Carlo is 3 to 5 '
        'times worse than kriging, polynomial chaos and a neural network, which are tied near the reference noise floor.')
picture(s, 'figs/deck_drivers.png', Inches(6.75), Inches(2.0), Inches(5.67), Inches(4.25),
        'Leading sensitivity drivers for reflection, coupling and isolation from 45 real runs with 90 percent bootstrap '
        'intervals. Board thickness and copper widths A7 and A3 dominate. All nine intervals contain the value from all 540 runs.')
takeaway(s, 'At equal runs, conventional Monte Carlo is 3 to 5 times less accurate; the three surrogates tie.')
notes(s, "Left: reference is Monte Carlo on all 540 real runs; errors near 0.017 sit at that reference's own noise floor. "
         'Right: intervals shown for the three leading drivers only – near-zero indices give an unreliable bootstrap. '
         'Be upfront: our method ties polynomial chaos and a neural network here. The win is against conventional Monte Carlo, '
         'which is what the criterion asks for, and the closed-form indices remove sampling noise entirely.')

# ==================================================================== slide 5  — why better: KPI 1, design cost (was slide 4)
s = S[3]
title(s, 'Title 1', 'KPI 1 – design search: conventional versus this pipeline', 28)
for name, text in (('Text Placeholder 2', 'Conventional: solver in the loop'),
                   ('Text Placeholder 4', 'This pipeline: one 60-run surrogate')):
    h = ph(s, name)
    h.text_frame.clear()
    set_text(h, [text], 18, bold=True, color=CHAR)
drop(ph(s, 'Content Placeholder 3'))
drop(ph(s, 'Content Placeholder 5'))
picture(s, 'figs/deck_conv.png', Inches(0.92), Inches(2.74), Inches(5.64), Inches(3.55),
        'Conventional cost at 30 minutes per simulation: which errors matter 277 days, how many boards pass 12 days, '
        'a design search of 1.92 million evaluations 110 years.')
picture(s, 'figs/deck_ours.png', Inches(6.75), Inches(2.74), Inches(5.64), Inches(3.55),
        'This pipeline: which errors matter 15 hours, how many boards pass 20 hours, a design search of 1.92 million '
        'surrogate evaluations 30 hours plus 5 minutes, all from one model built on 60 simulations.')
takeaway(s, 'Serial times at 30 minutes per run. With 100 solver licences in parallel, the design search alone still takes about a year.',
         top=6.4, size=14)
notes(s, 'Criterion 1 is the third row: 961 design points, each scored on 2,000 virtual manufacturing realizations, so 1.92 million '
         'surrogate evaluations in 5 minutes on a laptop. Say plainly that these are surrogate evaluations, and give the parallel caveat first.')

# ==================================================================== slide 6  — validation (was slide 5)
s = S[4]
title(s, 'Title 1', 'Validation: from 30% to 85% of boards passing', 26)
body = ph(s, 'Text Placeholder 3')
body.text_frame.clear()
set_text(body, ["Each scenario's real boards were held out, the model retrained, and its prediction compared with them: 5 of 5 inside the real interval.",
                'Last step: 85% predicted, 73% observed on 30 boards – consistent, not proof.',
                'Pass/fail agreement stays 92-97% when the illustrative limits are halved or tripled.',
                'Open: model bias, the confidential stack-up, no extra solver runs.'],
         14, space_after=10)
add_para(body, '0 of 734 million predicted matrices break passivity.',
         15, bold=True, color=BRAND, space_before=10)
drop(ph(s, 'Picture Placeholder 2'))
picture(s, 'figs/deck_stair.png', Inches(5.67), Inches(1.08), Inches(6.75), Inches(5.33),
        'Bar chart of boards passing under stated limits, from the full-curve model retrained without each scenario\'s boards: today 30 percent, '
        'new laminate 47, plus copper width A7 tolerance halved 57, plus board thickness tolerance halved 65, plus copper width A3 '
        'tolerance halved 85. The real boards in each scenario give 28, 43, 54, 63 and 73 percent with intervals that contain every prediction.')
notes(s, 'Pass rates depend on limits we chose; the mentor confirmed the challenge is not about board performance, so the message is the '
         'method and its checks, not the number itself. Sources of uncertainty are identified, not added together: sampling of the real '
         'boards (about 6 to 15 points), training-set variation (about 1 point), and model bias, which the data cannot resolve. '
         'The last bar has 30 real boards; the 11-point gap sits inside its 95% interval, so call it consistent, not a confirmation. '
         'The physics check ran on the full 3,012-output model: every one of 733.7 million predicted S-matrices, across every full-curve '
         'study (leave-window-out, 5-fold, scenario staircase, 961-point design grid), stayed passive.')

# ==================================================================== slide 7  — state of the art & references (NEW)
s_sota = prs.slides.add_slide(LAYOUTS['Two Content'])
title(s_sota, 'Title 1', 'State of the art, and where this differs', 28)
drop(ph(s_sota, 'Content Placeholder 2'))
drop(ph(s_sota, 'Picture Placeholder 3'))
picture(s_sota, 'figs/deck_sota.png', Inches(0.92), Inches(1.75), Inches(11.5), Inches(3.55),
        'Comparison table, common practice versus this project. Surrogate: kriging is used in about a third of RF surrogate studies; '
        'here it is checked against polynomial chaos and a neural network. Output: the literature compresses the response first; here PCA '
        'raised the error 2.8 times, so all 3,012 outputs are predicted. Physics: physics-informed networks need field data; only port data '
        'exists, so reciprocity is built in and passivity checked on 734 million matrices. Sensitivity: closed-form GP Sobol indices are known; '
        'here applied to 11 manufacturing tolerances, top drivers right from 30 runs. Yield: GP-assisted Monte Carlo re-checks boards in the '
        'solver; here no extra runs were allowed, so pass rates are checked against held-out real boards.')
citations = [['[1] Oakley & O\'Hagan (2004), J. R. Stat. Soc. B 66(3)',
              '[2] Marrel et al. (2009), Reliab. Eng. Syst. Saf. 94(3)',
              '[3] Le Gratiet, Cannamela & Iooss (2014), SIAM/ASA J. Uncertain. Quantif.',
              '[4] Systematic review, surrogate-based EM design of RF components (2026), Sensors 26(8):2504'],
             ['[5] Kersaudy et al. (2015), J. Comput. Phys. 286:103-117',
              '[6] Khan & Lowther (2022), IEEE Trans. Magn. 58(9)',
              '[7] Fuhrländer & Schöps (2020), J. Math. Ind. 10']]
for col, (x, w) in zip(citations, ((0.92, 6.3), (7.45, 4.97))):
    refs = s_sota.shapes.add_textbox(Inches(x), Inches(5.5), Inches(w), Inches(1.2))
    rtf = refs.text_frame
    rtf.word_wrap = True
    rtf.margin_left = rtf.margin_right = 0
    for i, c in enumerate(col):
        p2 = rtf.paragraphs[0] if i == 0 else rtf.add_paragraph()
        p2.space_after = Pt(2)
        r2 = p2.add_run()
        r2.text = c
        r2.font.size = Pt(10)
        r2.font.color.rgb = MUTED
notes(s_sota, 'This slide is the one Huawei\'s 2026-09-22 e-mail explicitly allows in addition to the 5-slide core. '
      'Keep it brief if asked live: the one-line version is "kriging is already the field\'s default for EM surrogates; '
      'the difference here is enforcing measurement-level physics because no field data existed, and checking every choice '
      'against this specific dataset rather than assuming the literature default (PCA) would transfer."')

# ==================================================================== slide 8  — project folder / QR (was slide 6)
s = S[5]
title(s, 'Title 1', 'Project folder', 28)
body = ph(s, 'Text Placeholder 3')
body.text_frame.clear()
set_text(body, ['Scan to open the full material',
                'README: pipeline, evidence and limitations',
                'Code that reproduces every number',
                'Figures and the architecture report'], 16, space_after=10)
drop(ph(s, 'Picture Placeholder 2'))
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.29), Inches(1.64), Inches(4.5), Inches(4.16))
box.name = 'QR_PLACEHOLDER'
box.fill.background()
box.line.color.rgb = MUTED
box.line.width = Pt(2)
box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
tf = box.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = 'QR code goes here'
r.font.size = Pt(20)
r.font.bold = True
r.font.color.rgb = MUTED
p2 = tf.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run()
r2.text = 'run insert_qr.py with your folder link'
r2.font.size = Pt(14)
r2.font.color.rgb = MUTED
notes(s, 'The template reserves this slide for a link to a folder. Share the code, README and report there. '
         'Do not put the Huawei dataset in it.')

# ==================================================================== slide 9  — thank you / call to action (was slide 7)
s = S[6]
title(s, 'Title 1', 'THANK YOU', 40)
cta = s.shapes.add_textbox(Inches(1.2), Inches(5.35), Inches(10.93), Inches(1.0))
cta.text_frame.word_wrap = True
p = cta.text_frame.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = 'From 277 days of solver time to 15 hours – with the physics checked on every prediction.'
r.font.size = Pt(16)
r.font.bold = True
r.font.color.rgb = BRAND
p2 = cta.text_frame.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run()
r2.text = '60 solver runs  ·  exact sensitivity analysis  ·  pass rates confirmed on held-out real boards'
r2.font.size = Pt(13)
r2.font.color.rgb = MUTED
notes(s, 'Close on the one line to remember: 277 days of solver time become 15 hours, and every one of 734 million predicted '
         'matrices obeys the physics. Then stop -- do not trail off into a plain thank-you.')

# slides added from a layout lack the footer and slide number the template's own slides carry: copy them over
for new in (s_problem, s_sota):
    for shp in S[1].shapes:
        if shp.is_placeholder and shp.name.startswith(('Footer', 'Slide Number')):
            new.shapes._spTree.append(copy.deepcopy(shp._element))

# ==================================================================== reorder: title, problem, solution, kpi2, kpi1, validation, sota, qr, ending
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
# current append order in ids: [title, solution, kpi2, kpi1, validation, qr, ending, problem, sota]
order = [0, 7, 1, 2, 3, 4, 8, 5, 6]
new_ids = [ids[i] for i in order]
for el in ids:
    sldIdLst.remove(el)
for el in new_ids:
    sldIdLst.append(el)

for sl in prs.slides:
    footer(sl)
prs.save(OUT)
print('saved', OUT, 'slides:', len(prs.slides))
