"""Figures drawn from the supplied dataset itself."""
import numpy as np, matplotlib.pyplot as plt
from figs_core import *
import load_huawei as L
P,f,S=L.load()
db=lambda x:20*np.log10(np.clip(np.abs(x),1e-12,None))

# 6. one design across frequency ---------------------------------------------
fig,ax=plt.subplots(figsize=(11,4.8))
k=0
ax.plot(f,db(S[k,:,1,0]),color=TEAL,lw=2.8,label='S21  through path, port 1 to 2')
ax.plot(f,db(S[k,:,2,0]),color=COPPER,lw=2.8,label='S31  tapped path, port 1 to 3')
ax.plot(f,db(S[k,:,0,0]),color=INK,lw=2.8,label='S11  reflected back out of port 1')
i=np.argmin(db(S[k,:,0,0]))
ax.annotate('resonance:\nreflection is lowest here',xy=(f[i],db(S[k,i,0,0])),
            xytext=(f[i]+0.055,db(S[k,i,0,0])+7),fontsize=10.5,color=INK,
            arrowprops=dict(arrowstyle='->',color=INK,lw=1.6))
ax.set_xlabel('frequency in GHz',fontsize=11.5); ax.set_ylabel('decibels',fontsize=11.5)
ax.legend(frameon=False,fontsize=10.5,loc='lower right')
ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color=LIGHT,lw=1.2)
ax.set_title('One board, measured at all 251 frequencies. This is one design in your file.',
             fontsize=12.5,loc='left',pad=12)
fig.tight_layout(); save(fig,'06_sweep')

# 7. what manufacturing scatter does -----------------------------------------
fig,axes=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.5,1]})
a=axes[0]
for k in range(0,540,3):
    a.plot(f,db(S[k,:,0,0]),color=TEAL,lw=0.5,alpha=0.20)
a.plot(f,db(S[:,:,0,0]).mean(0)*0+np.median(db(S[:,:,0,0]),0),color=INK,lw=2.6)
a.set_xlabel('frequency in GHz'); a.set_ylabel('reflection S11, dB')
a.spines[['top','right']].set_visible(False)
a.set_title('Every one of the 540 boards, reflection',fontsize=11.5,loc='left')
a.text(1.205,-27,'each faint line is one manufactured board\nthe dark line is the typical one',
       fontsize=9.6,color=GREY,va='bottom')
b=axes[1]
w=db(S[:,:,0,0]).max(1)
b.hist(w,bins=30,color=COPPER,alpha=0.85)
b.axvline(np.median(w),color=INK,lw=2.2)
b.set_xlabel('worst reflection in band, dB'); b.set_ylabel('number of boards')
b.spines[['top','right']].set_visible(False)
b.set_title('The spread that causes rejects',fontsize=11.5,loc='left')
fig.suptitle('Identical drawings, different boards: this is what manufacturing tolerance means',
             fontsize=13,y=1.03)
fig.tight_layout(); save(fig,'07_spread')

# 10. sensitivity --------------------------------------------------------------
import json
k=json.load(open('results_kpi.json'))
st=np.array(k['sobol']['S11_max_dB']['ST']); nm=k['param_names']
o=np.argsort(st)
pretty={'$DK':'material constant','$R0402_1':'resistor 1','$R0402_2':'resistor 2',
        '$R0402_RL':'load resistor','TOL_LW_A1':'copper width A1','TOL_LW_A3':'copper width A3',
        'TOL_LW_A5':'copper width A5','TOL_LW_A7':'copper width A7','TOL_LW_A9':'copper width A9',
        'TOL_h':'board thickness','t_art1':'copper thickness'}
fig,ax=plt.subplots(figsize=(11,5.0))
cols=[COPPER if st[i]>0.09 else GREY for i in o]
ax.barh([pretty[nm[i]] for i in o],st[o],color=cols,height=0.66)
for i,v in zip(o,st[o]):
    ax.text(v+0.012,pretty[nm[i]],f'{v:.3f}',va='center',fontsize=10.2,color=INK)
ax.set_xlim(0,0.62); ax.set_xlabel('share of the variation it explains',fontsize=11)
ax.spines[['top','right']].set_visible(False); ax.set_xticks([])
ax.set_title('Which manufacturing errors actually matter, for reflection',
             fontsize=12.5,loc='left',pad=12)
ax.text(0.30,0.6,'These two explain 89%.\nTighten them.',fontsize=11.5,color=COPPER,fontweight='bold')
ax.text(0.30,7.2,'These nine barely register.\nStop paying to control them.',fontsize=11,color=GREY)
fig.tight_layout(); save(fig,'10_sensitivity')

# 14. how little data is needed ------------------------------------------------
fig,ax=plt.subplots(figsize=(10.5,4.6))
n=[60,120,240,400]
ax.plot(n,[0.0435,0.0313,0.0220,0.0145],'o-',color=GREY,lw=2.2,ms=7,label='with my compression step')
ax.plot([60,120],[0.0160,0.0148],'o-',color=COPPER,lw=3.2,ms=9,label='without it  (the right way)')
ax.axhline(0.0160,color=COPPER,ls=':',lw=1.6)
ax.annotate('60 designs here beat\n400 designs there',xy=(120,0.0148),xytext=(175,0.0205),
            fontsize=10.8,color=COPPER,arrowprops=dict(arrowstyle='->',color=COPPER,lw=1.8))
ax.set_yscale('log'); ax.set_xlabel('number of simulated designs used for training',fontsize=11.5)
ax.set_ylabel('prediction error',fontsize=11.5)
ax.set_yticks([0.015,0.02,0.03,0.05]); ax.set_yticklabels(['1.5%','2%','3%','5%'])
ax.legend(frameon=False,fontsize=10.5); ax.spines[['top','right']].set_visible(False)
ax.set_title('You do not need all 540 simulations',fontsize=12.5,loc='left',pad=12)
fig.tight_layout(); save(fig,'14_dataeff')

# 11/12. yield and redesign ----------------------------------------------------
fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
for a,(before,after,t) in zip(axes,[((39.5,60.5),(87.5,12.5),'')]*1+[(None,None,None)]*0,['']*1):
    pass
a=axes[0]
a.barh([''],[39.5],color=GOOD,height=0.5); a.barh([''],[60.5],left=[39.5],color='#E3E8EB',height=0.5)
a.text(19,0,'39.5% pass',ha='center',va='center',color=WHITE,fontweight='bold',fontsize=12)
a.text(70,0,'60.5% rejected',ha='center',va='center',color=GREY,fontsize=11)
a.set_title('Today',fontsize=12,loc='left'); a.axis('off')
b=axes[1]
b.barh([''],[87.5],color=GOOD,height=0.5); b.barh([''],[12.5],left=[87.5],color='#E3E8EB',height=0.5)
b.text(44,0,'87.5% pass',ha='center',va='center',color=WHITE,fontweight='bold',fontsize=12)
b.text(94,0,'12.5%',ha='center',va='center',color=GREY,fontsize=11)
b.set_title('After the model redesigns it',fontsize=12,loc='left'); b.axis('off')
fig.suptitle('Out of 100 boards made, how many meet specification',fontsize=13,y=0.99)
fig.text(0.5,0.04,'8 million candidate designs tested in 150 seconds, with no extra simulator runs. '
         'Still to be confirmed in the real simulator.',ha='center',fontsize=10,color=GREY)
save(fig,'12_yield')
