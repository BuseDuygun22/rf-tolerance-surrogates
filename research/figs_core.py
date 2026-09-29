"""Figure library for the teaching deck and the architecture report.
Real measured curves come from the supplied dataset wherever possible."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle
import load_huawei as L

INK='#12263A'; TEAL='#1C7293'; COPPER='#C97B26'; GOOD='#2F7D4F'
RED='#A8392F'; LIGHT='#EDF2F5'; GREY='#7A8B96'; WHITE='#FFFFFF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.edgecolor':GREY,'axes.labelcolor':INK,
                     'xtick.color':INK,'ytick.color':INK,'text.color':INK,
                     'figure.facecolor':WHITE,'savefig.facecolor':WHITE})
DPI=200
def save(fig,name):
    fig.savefig(f'figs/{name}.png',dpi=DPI,bbox_inches='tight',pad_inches=0.15)
    plt.close(fig); print('figs/'+name+'.png')

def blank(w,h):
    fig=plt.figure(figsize=(w,h))
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100*h/w)
    ax.axis('off'); return fig,ax

def box(ax,x,y,w,h,label,sub=None,fc=WHITE,ec=INK,tc=INK,fs=13,lw=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.6,rounding_size=1.2',
                 fc=fc,ec=ec,lw=lw,zorder=2))
    ax.text(x+w/2,y+h/2+(1.6 if sub else 0),label,ha='center',va='center',
            fontsize=fs,fontweight='bold',color=tc,zorder=3)
    if sub: ax.text(x+w/2,y+h/2-3.0,sub,ha='center',va='center',fontsize=fs-3.5,color=tc,zorder=3)

def arrow(ax,x1,y1,x2,y2,c=INK,lw=2.4,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=18,
                 lw=lw,color=c,linestyle=ls,zorder=4,shrinkA=0,shrinkB=0))
