"""Concept and architecture figures."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from figs_core import *

# 8. the slow way vs the fast way ---------------------------------------------
fig,ax=blank(12,6.2)
ax.text(4,48,'THE SLOW WAY',fontsize=13,fontweight='bold',color=RED)
for i,(lab,sub) in enumerate([('geometry','+ materials'),('Maxwell','equations'),
                              ('fields','everywhere'),('S-parameters',None)]):
    box(ax,4+i*23,34,17,10,lab,sub,fc=WHITE,ec=RED if i<3 else INK,fs=11.5)
    if i<3: arrow(ax,21+i*23,39,27+i*23,39,RED,2.2)
ax.text(50,29.5,'minutes per design',ha='center',fontsize=11.5,color=RED,style='italic')
ax.text(4,20,'THE FAST WAY',fontsize=13,fontweight='bold',color=GOOD)
for i,(lab,sub) in enumerate([('geometry','+ materials'),('trained','model'),('S-parameters',None)]):
    box(ax,4+i*23,6,17,10,lab,sub,fc=WHITE,ec=GOOD if i<2 else INK,fs=11.5)
    if i<2: arrow(ax,21+i*23,11,27+i*23,11,GOOD,2.2)
ax.text(40,1.5,'0.03 milliseconds per design',ha='center',fontsize=11.5,color=GOOD,style='italic')
ax.text(78,12,'The model never solves\nMaxwell. It learned the\npattern from 540 examples\nthat did.',
        fontsize=11,va='center',color=INK)
save(fig,'08_slow_fast')

# 9. the architecture ----------------------------------------------------------
fig,ax=blank(13,7.4)
ax.text(2,54,'ONCE, UP FRONT',fontsize=11,fontweight='bold',color=GREY)
box(ax,2,42,20,9,'60 designs','simulated in CST',fc=LIGHT,fs=11)
arrow(ax,22,46.5,28,46.5,INK,2.2)
box(ax,28,42,22,9,'TRAIN THE MODEL','once, about 5 min',fc=WHITE,ec=TEAL,fs=11)
ax.text(2,36,'PHYSICS BUILT IN',fontsize=11,fontweight='bold',color=COPPER)
for i,(t_,s_) in enumerate([('reciprocity','6 outputs, not 9'),
                            ('passivity','checked, not forced'),
                            ('smoothness','why 60 is enough')]):
    box(ax,2+i*17,26,15,8,t_,s_,fc='#FBF2E7',ec=COPPER,fs=10.5)
arrow(ax,50,46.5,56,46.5,INK,2.2)
box(ax,56,40,26,13,'FAST SURROGATE','answers in 0.03 ms',fc=WHITE,ec=TEAL,lw=3,fs=13)
ax.text(2,20,'THEN, MILLIONS OF TIMES',fontsize=11,fontweight='bold',color=GREY)
outs=[('WHICH ERRORS\nMATTER','53,000 runs\n1.7 seconds',GOOD),
      ('HOW MANY\nBOARDS PASS','100,000 runs\n3.2 seconds',GOOD),
      ('REDESIGN FOR\nHIGHER YIELD','8,000,000 runs\n150 seconds',COPPER)]
for i,(t_,s_,c) in enumerate(outs):
    box(ax,4+i*30,4,24,12,t_,s_,fc=WHITE,ec=c,fs=11)
    arrow(ax,69-((69-(16+i*30))*0.0),38,16+i*30,17,c,1.8,ls='--')
save(fig,'09_architecture')

# 13. three levels of physics-informed ----------------------------------------
fig,ax=blank(12,6.6)
lv=[('LEVEL 1','Physics-blind','numbers in, numbers out.\nNo physical knowledge at all.',GREY,'#F2F4F5'),
    ('LEVEL 2','Physics-constrained','the prediction is forced to obey\nthe laws the measurement obeys:\nreciprocity, passivity, smoothness.',COPPER,'#FBF2E7'),
    ('LEVEL 3','Full PINN','the network predicts the fields and is\npenalised for breaking Maxwell at\nevery point in space.',TEAL,'#EAF2F5')]
for i,(n_,t_,d_,c,bg) in enumerate(lv):
    y=38-i*17
    ax.add_patch(FancyBboxPatch((3,y),94,14,boxstyle='round,pad=0.5,rounding_size=1',
                 fc=bg,ec=c,lw=2.4))
    ax.text(7,y+9.6,n_,fontsize=10,fontweight='bold',color=c)
    ax.text(7,y+4.6,t_,fontsize=14,fontweight='bold',color=INK)
    ax.text(38,y+7,d_,fontsize=10.8,va='center',color=INK)
ax.text(78,45,'← what most\n   people do',fontsize=10.5,color=GREY,va='center')
ax.text(78,28,'← what your data\n   supports',fontsize=11,color=COPPER,va='center',fontweight='bold')
ax.text(78,11,'← needs field data\n   you do not have',fontsize=10.5,color=TEAL,va='center')
ax.text(3,56,'"Physics-informed" is not one thing. It has levels.',fontsize=13.5,color=INK)
save(fig,'13_levels')

# 15. what the criteria reward -------------------------------------------------
fig,ax=blank(12,6.4)
rows=[('CRITERION 1','Accelerated design vs\nconventional engineering',
       'Redesign for yield:\n8 M candidates in 150 s',COPPER),
      ('CRITERION 2','Sensitivity analysis vs\nconventional Monte Carlo',
       '153,000 cases in 18.6 s\nagainst 106 days of solver',GOOD),
      ('STATED CHALLENGE','How much training data\nthe model needs',
       '60 of the 540 designs,\nabout one ninth',TEAL)]
for i,(a_,b_,c_,col) in enumerate(rows):
    y=36-i*17
    ax.add_patch(FancyBboxPatch((2,y),41,14,boxstyle='round,pad=0.5,rounding_size=1',
                 fc=LIGHT,ec=GREY,lw=1.4))
    ax.text(5,y+10,a_,fontsize=9.5,fontweight='bold',color=GREY)
    ax.text(5,y+4.5,b_,fontsize=12,color=INK,va='center')
    arrow(ax,44,y+7,52,y+7,col,2.6)
    ax.add_patch(FancyBboxPatch((53,y),44,14,boxstyle='round,pad=0.5,rounding_size=1',
                 fc=WHITE,ec=col,lw=2.4))
    ax.text(56,y+7,c_,fontsize=12,color=INK,va='center',fontweight='bold')
ax.text(2,55,'What the marking scheme asks for, and what answers it',fontsize=13.5,color=INK)
ax.text(2,50,'Both criteria are ratios. You hold one half. Huawei holds the other.',
        fontsize=11,color=RED,style='italic')
save(fig,'15_criteria')
