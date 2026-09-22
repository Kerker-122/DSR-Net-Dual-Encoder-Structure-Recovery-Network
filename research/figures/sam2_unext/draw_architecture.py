"""Source-faithful SAM2-UNeXT schematic; no model execution or synthetic results."""
from pathlib import Path
import os
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
sys.path.insert(0, str(ROOT / '.agents/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
                     'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.linewidth': .7, 'savefig.facecolor': 'white'})
INK = '#293743'
BLUE = '#E4EFF8'
BLUE_EDGE = '#5A7F9D'
PURPLE = '#EEE8F5'
PURPLE_EDGE = '#89719F'
TEAL = '#E3F0EC'
TEAL_EDGE = '#508778'
SAND = '#F7ECDD'
SAND_EDGE = '#AE8855'

fig = plt.figure(figsize=(7.0866141732, 6.0236220472))  # 180 x 153 mm
gs = fig.add_gridspec(2, 2, height_ratios=[3.7, 1], left=.025, right=.975,
                      bottom=.035, top=.975, hspace=.18, wspace=.09)
a = fig.add_subplot(gs[0, :])
b = fig.add_subplot(gs[1, 0])
c = fig.add_subplot(gs[1, 1])
for ax, limits in [(a,(180,116)), (b,(86,30)), (c,(86,30))]:
    ax.set_xlim(0,limits[0]); ax.set_ylim(0,limits[1]); ax.axis('off')

def txt(ax, x,y,t,size=7,weight='normal',ha='center',color=INK):
    return ax.text(x,y,t,ha=ha,va='center',fontsize=size,fontweight=weight,
                   color=color,linespacing=1.35)

def box(ax,x,y,w,h,t,fc=TEAL,ec=TEAL_EDGE,size=7,dashed=False):
    ax.add_patch(FancyBboxPatch((x,y-h/2),w,h,
                 boxstyle='round,pad=0.0,rounding_size=1.2',
                 linewidth=.75,edgecolor=ec,facecolor=fc,
                 linestyle=(0,(3,2)) if dashed else '-'))
    txt(ax,x+w/2,y,t,size)

def arrow(ax,pts,color=INK,dashed=False,head=True,lw=.8):
    for p,q in zip(pts[:-2],pts[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,
                linestyle=(0,(3,2)) if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>' if head else '-',
                 mutation_scale=6,linewidth=lw,color=color,
                 linestyle=(0,(3,2)) if dashed else '-',shrinkA=0,shrinkB=0,zorder=1))

def circ(ax,x,y,label):
    ax.add_patch(Circle((x,y),2.1,facecolor='white',edgecolor=INK,lw=.7,zorder=3))
    txt(ax,x,y,label,6.5)

# a: the complete forward path, including the single shared DINO feature.
txt(a,0,113,'a',10,'bold',ha='left')
txt(a,6,113,'SAM2-UNeXT architecture',9,'bold',ha='left')
txt(a,44,98.5,'SAM2 Hiera-L',7.5,'bold')
txt(a,100,98.5,'Dense feature fusion',7.5,'bold')
txt(a,155,98.5,'Decoder',7.5,'bold')
txt(a,44,94,'Frozen blocks + trainable adapters',5.8)
txt(a,100,94,'One shared DINOv2 output',6)

rows=[85,65,45,25]
channels=[144,288,576,1152]
strides=[4,8,16,32]
box(a,1,85,19,12,'RGB input\nH × W × 3','#F3F5F7','#8B979F',6.5)
arrow(a,[(20,85),(29,85)])
arrow(a,[(24,85),(24,7),(29,7)])
for i,(y,ch,st) in enumerate(zip(rows,channels,strides),1):
    box(a,29,y,30,12,f'Stage {i} + Adapter\n{ch} ch · H/{st} × W/{st}',BLUE,BLUE_EDGE,6.6)
    if i<4:
        arrow(a,[(44,y-6),(44,y-14)],BLUE_EDGE)
    arrow(a,[(59,y),(109.9,y)],BLUE_EDGE)
    circ(a,112,y,'C')
    box(a,119,y,20,10,'1 × 1 Conv\n128 ch',TEAL,TEAL_EDGE,6.5)
    arrow(a,[(114.1,y),(119,y)])
    box(a,151,y,27,12,f'Up{i}\n128 ch · H/{st//2}',SAND,SAND_EDGE,6.6)
    arrow(a,[(139,y),(151,y)])
    if i<4:
        arrow(a,[(164.5,y-14),(164.5,y-6)],SAND_EDGE)
    # Projection precedes spatial resizing, exactly as in forward().
    ay=y-8
    box(a,78,ay,25,8,f'1 × 1: 1024 → {ch}\nResize to H/{st} × W/{st}',PURPLE,PURPLE_EDGE,5.9)
    arrow(a,[(70,ay),(78,ay)],PURPLE_EDGE,True)
    a.add_patch(Circle((70,ay),.45,facecolor=PURPLE_EDGE,edgecolor='none'))
    arrow(a,[(103,ay),(112,ay),(112,y-2.1)],PURPLE_EDGE,True)

# A single auxiliary feature is shared across all four projections.
box(a,29,7,30,10,'Resize 448 × 448\nDINOv2 ViT-L/14',PURPLE,PURPLE_EDGE,6.6,True)
arrow(a,[(59,7),(70,7),(70,77)],PURPLE_EDGE,True,False)
txt(a,94,7,'Last feature only\n1024 × 32 × 32',6.5,color=PURPLE_EDGE)
txt(a,144,7,'Dashed outline: frozen module\nC: channel concatenation',5.9)

# Full-resolution logits; the code does not apply a sigmoid here.
arrow(a,[(164.5,91),(164.5,107),(151,107)])
box(a,130,107,21,8,'1 × 1 Conv\n128 → 1',SAND,SAND_EDGE,6.3)
arrow(a,[(130,107),(122,107)])
box(a,102,107,20,8,'Resize ×2',SAND,SAND_EDGE,6.5)
arrow(a,[(102,107),(93,107)])
txt(a,76,107,'Output logits\nH × W × 1',6.5)

# b: block input adaptation; pre-block residual, not post-block adaptation.
txt(b,0,28.5,'b',10,'bold',ha='left')
txt(b,6,28.5,'Trainable Adapter',8,'bold',ha='left')
txt(b,2,12,'X',7)
box(b,10,17,37,9,'Linear C → 32 → C\nGELU after each linear',TEAL,TEAL_EDGE,6.5)
arrow(b,[(4,12),(7,12),(7,17),(10,17)])
arrow(b,[(4,12),(7,12),(7,6),(53,6),(53,9.9)])
arrow(b,[(47,17),(53,17),(53,14.1)])
circ(b,53,12,'+')
arrow(b,[(55.1,12),(61,12)])
box(b,61,12,25,12,'Frozen\nHiera block',BLUE,BLUE_EDGE,6.5,True)
txt(b,43,1.2,'Applied before every Hiera block',6)

# c: Up.forward concatenates at its input resolution and then upsamples.
txt(c,0,28.5,'c',10,'bold',ha='left')
txt(c,6,28.5,'Decoder Up block',8,'bold',ha='left')
txt(c,7,18,'Input',6.5)
txt(c,7,7,'Skip',6.5)
arrow(c,[(14,18),(19,18),(19,13),(22.9,13)])
arrow(c,[(14,7),(25,7),(25,10.9)])
circ(c,25,13,'C')
arrow(c,[(27.1,13),(33,13)])
box(c,33,13,20,10,'Bilinear\nupsample ×2',SAND,SAND_EDGE,6.2)
arrow(c,[(53,13),(59,13)])
box(c,59,13,27,12,'[3 × 3 Conv\n+ BN + ReLU] ×2',SAND,SAND_EDGE,6.1)
txt(c,43,1.2,'Up4 has no skip input; Up3–Up1 concatenate',5.9)

fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=OUT/'sam2_unext.alignment.json',
                                   overlay_svg=OUT/'sam2_unext.alignment.svg',strict=True)
for ext in ['pdf','svg']:
    fig.savefig(OUT/f'sam2_unext_architecture.{ext}')
fig.savefig(OUT/'sam2_unext_architecture.png',dpi=600)
fig.savefig(OUT/'sam2_unext_architecture.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(OUT/'sam2_unext_preview.png',dpi=300)
source=ROOT/'SAM2UNeXT.py'
(OUT/'source_manifest.json').write_text(json.dumps({
    'source':'SAM2UNeXT.py','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'figure_mm':[180,153],'dpi_png':600,'model_executed':False,
    'notes':['Nominal spatial sizes assume compatible H,W.',
             'No probability sigmoid in model return path.',
             'Redundant first self.head(x) assignment omitted as dead output.',
             'Padding alignment within Up is omitted from the overview.',
             'Frozen status describes backbone parameters; adapters are trainable.']
},indent=2),encoding='utf-8')
print(OUT/'sam2_unext_architecture.pdf')
