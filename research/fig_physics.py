"""Slides 1-8: the physics, from zero."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from figs_core import *

# 1. the three-port component ------------------------------------------------
fig,ax=blank(12,6)
box(ax,34,18,30,22,'DIRECTIONAL','COUPLER',fc=LIGHT,ec=INK,fs=16)
arrow(ax,6,29,33,29,TEAL,3.2)
ax.text(6,33,'PORT 1',fontsize=12,fontweight='bold',color=TEAL)
ax.text(6,24.5,'signal goes in',fontsize=10.5,color=GREY)
arrow(ax,65,29,94,29,TEAL,3.2)
ax.text(76,33,'PORT 2',fontsize=12,fontweight='bold',color=TEAL)
ax.text(70,24.5,'57% of the power carries on',fontsize=10.5,color=GREY)
arrow(ax,49,17,49,4,COPPER,3.2)
ax.text(52,10,'PORT 3',fontsize=12,fontweight='bold',color=COPPER)
ax.text(52,6,'5% is tapped off to be measured',fontsize=10.5,color=GREY)
ax.text(50,46,'A tap: it samples the signal without interrupting it',
        ha='center',fontsize=13.5,style='italic',color=INK)
save(fig,'01_coupler')

# 2. amplitude and phase ------------------------------------------------------
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
t=np.linspace(0,4*np.pi,600)
a=axes[0]
a.plot(t,np.sin(t),color=TEAL,lw=2.6,label='small amplitude')
a.plot(t,2*np.sin(t),color=COPPER,lw=2.6,label='large amplitude')
a.set_title('AMPLITUDE  =  how big the wave is',fontsize=12,fontweight='bold',pad=12)
a.legend(frameon=False,fontsize=10,loc='upper right')
b=axes[1]
b.plot(t,np.sin(t),color=TEAL,lw=2.6,label='reference wave')
b.plot(t,np.sin(t-1.4),color=COPPER,lw=2.6,label='same size, shifted')
b.annotate('',xy=(np.pi/2,1.16),xytext=(np.pi/2+1.4,1.16),
           arrowprops=dict(arrowstyle='<->',color=INK,lw=1.8))
b.text(np.pi/2+0.7,1.35,'phase shift',ha='center',fontsize=10.5,color=INK)
b.set_title('PHASE  =  where it is in its cycle',fontsize=12,fontweight='bold',pad=12)
b.legend(frameon=False,fontsize=10,loc='lower right')
for a_ in axes:
    a_.set_ylim(-2.5,1.9); a_.set_xticks([]); a_.set_yticks([])
    a_.spines[['top','right','left']].set_visible(False)
    a_.axhline(0,color=GREY,lw=1)
fig.suptitle('A radio signal is a wave with exactly two properties that matter',
             fontsize=13.5,y=1.04)
fig.tight_layout(); save(fig,'02_wave')

# 3. why real and imaginary ---------------------------------------------------
fig=plt.figure(figsize=(11,4.8)); ax=fig.add_axes([0.06,0.08,0.40,0.86])
ax.axhline(0,color=GREY,lw=1.2); ax.axvline(0,color=GREY,lw=1.2)
a_,b_=1.5,1.1
ax.add_patch(FancyArrowPatch((0,0),(a_,b_),arrowstyle='-|>',mutation_scale=20,lw=3,color=COPPER))
ax.plot([a_,a_],[0,b_],ls=':',color=GREY,lw=1.6); ax.plot([0,a_],[b_,b_],ls=':',color=GREY,lw=1.6)
ax.text(a_/2,-0.22,'real part',ha='center',fontsize=11,color=TEAL)
ax.text(-0.13,b_/2,'imaginary\npart',ha='right',va='center',fontsize=11,color=TEAL)
ax.text(0.62,0.83,'length = amplitude',fontsize=10.5,color=COPPER,rotation=36)
th=np.linspace(0,np.arctan2(b_,a_),40)
ax.plot(0.42*np.cos(th),0.42*np.sin(th),color=INK,lw=1.6)
ax.text(0.52,0.16,'angle = phase',fontsize=10.5,color=INK)
ax.set_xlim(-0.6,2.1); ax.set_ylim(-0.5,1.6); ax.set_xticks([]); ax.set_yticks([])
ax.spines[:].set_visible(False)
ax2=fig.add_axes([0.52,0.08,0.44,0.86]); ax2.axis('off')
ax2.text(0,0.94,'Why your file has two columns per measurement',fontsize=13,fontweight='bold',va='top')
lines=['One number cannot carry both amplitude and phase.',
       '','Engineers store the pair as a point on a plane.',
       'The horizontal coordinate is called the real part.',
       'The vertical one is called the imaginary part.',
       '','Nothing about it is imaginary. It is simply a',
       'second coordinate, and the name is historical.',
       '','So  S1,1_real  and  S1,1_imag  together tell you',
       'how much of the wave came back and how far it',
       'slipped in its cycle on the way.']
for i,t_ in enumerate(lines):
    ax2.text(0,0.80-i*0.068,t_,fontsize=11.2,va='top',
             color=INK if not t_.startswith('So ') else COPPER,
             fontweight='bold' if t_.startswith('So ') else 'normal')
save(fig,'03_complex')

# 4. the S-matrix and reciprocity --------------------------------------------
fig,ax=blank(11,6.2)
x0,y0,c=22,8,16
for i in range(3):
    for j in range(3):
        x=x0+j*c; y=y0+(2-i)*c
        diag=(i==j); mir=(i>j)
        fc = LIGHT if diag else ('#E8EEF1' if mir else WHITE)
        ax.add_patch(Rectangle((x,y),c-1.5,c-1.5,fc=fc,ec=INK if not mir else GREY,
                     lw=2 if not mir else 1.2,zorder=2))
        ax.text(x+(c-1.5)/2,y+(c-1.5)/2+2.4,f'S{i+1}{j+1}',ha='center',fontsize=15,
                fontweight='bold',color=INK if not mir else GREY,zorder=3)
        lab='bounces back' if diag else f'port {j+1} to {i+1}'
        ax.text(x+(c-1.5)/2,y+(c-1.5)/2-3.2,lab,ha='center',fontsize=8.6,
                color=INK if not mir else GREY,zorder=3)
ax.text(x0+1.5*c-0.7,y0+3*c+2.5,'OUT OF PORT  ↑    INTO PORT  →',ha='center',fontsize=10.5,color=GREY)
ax.plot([x0-1,x0+3*c-2.5],[y0+3*c-1.5,y0-1],color=COPPER,lw=2.4,ls='--',zorder=5)
ax.text(76,40,'RECIPROCITY',fontsize=13,fontweight='bold',color=COPPER)
ax.text(76,34.5,'The greyed half is a mirror\nof the white half.\n\nS12 = S21, S13 = S31, S23 = S32.\n\nTrue in your data to 11\ndecimal places, so the model\nonly has to learn 6 of the 9.',
        fontsize=10.6,va='top',color=INK)
save(fig,'04_smatrix')

# 5. decibels -----------------------------------------------------------------
fig,ax=plt.subplots(figsize=(11,4.6))
db=np.array([0,-3,-6,-10,-13,-20,-25,-30])
pw=10**(db/10)*100
cols=[GOOD if d>=-6 else (TEAL if d>=-13 else COPPER) for d in db]
bars=ax.barh(range(len(db)),pw,color=cols,height=0.62)
for i,(d,p) in enumerate(zip(db,pw)):
    ax.text(p+1.6,i,f'{p:.1f}% of the power survives',va='center',fontsize=10.6,color=INK)
ax.set_yticks(range(len(db))); ax.set_yticklabels([f'{d} dB' for d in db],fontsize=11.5,fontweight='bold')
ax.invert_yaxis(); ax.set_xlim(0,132); ax.set_xticks([])
ax.spines[['top','right','bottom']].set_visible(False)
ax.set_title('Decibels are just a squashed way of writing "what fraction got through"',
             fontsize=12.5,pad=14,loc='left')
ax.text(0,-1.15,'Your device: through path −2.4 dB (57% passes) · tap −12.9 dB (5%) · leak between ports −25 dB (0.3%)',
        fontsize=10.4,color=GREY,transform=ax.get_yaxis_transform(),clip_on=False)
fig.tight_layout(); save(fig,'05_db')
