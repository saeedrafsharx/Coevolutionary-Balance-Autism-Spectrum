"""Render Fig 1 with the manuscript's magenta/dark-green palette."""
from pathlib import Path
import hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
MAGENTA='#DA005D'
GREEN='#00473A'
def draw_workflow():
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.5,6.2),constrained_layout=True)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[
        (.5,.92,.88,.10,'Existing ABIDE I source pool: 182 (90 ASD / 92 controls)',MAGENTA),
        (.5,.76,.88,.12,'Age 18–30; observed FIQ >80 → 140 (69 ASD / 71 controls)\nFixed coverage-qualified CC200 network: 106 nodes',GREEN),
        (.25,.55,.43,.15,'Downloaded filtered ROI series\nGSR and no-GSR\nLocal Fisher-transformed FC',GREEN),
        (.75,.55,.43,.15,'Downloaded fALFF maps/masks\nUpstream computation\nLocal ROI means / relative states',MAGENTA),
        (.5,.34,.88,.12,'Static H; pairwise configurations; global/local efficiency\nWithin-network and between-network descriptors',MAGENTA),
        (.2,.11,.29,.15,'Null configurations\nAll 140 participants\nBoth diagnoses',GREEN),
        (.51,.11,.29,.15,'Adjusted group models\n136 (67 ASD / 69 controls)\nSite, age, FIQ and FD',MAGENTA),
        (.82,.11,.29,.15,'Clinical / prediction\nScore-wise missingness\nSite-held-out models',GREEN),
    ]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.006,rounding_size=.018',facecolor=color,edgecolor=color,lw=1))
        ax.text(x,y,label,ha='center',va='center',fontsize=9,color='white',linespacing=1.4)
    edges=[((.5,.865),(.5,.83),MAGENTA),((.35,.695),(.25,.635),GREEN),((.65,.695),(.75,.635),MAGENTA),((.25,.47),(.4,.41),GREEN),((.75,.47),(.6,.41),MAGENTA),((.35,.27),(.2,.195),GREEN),((.5,.27),(.51,.195),MAGENTA),((.65,.27),(.82,.195),GREEN)]
    for start,end,color in edges:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','color':color,'lw':1.5,'mutation_scale':12})
    ax.text(.5,.003,'Stage colors reuse the manuscript palette; they do not indicate diagnosis.',ha='center',va='bottom',fontsize=8,color='#333333')
    return fig

def main():
    root=Path(__file__).resolve().parents[1];out=root/'revision/resubmission'
    fig=draw_workflow()
    for ext in ['pdf','png']:
        fig.savefig(out/'figures'/('Fig1.'+ext),dpi=200,facecolor='white')
    path=out/'upload_figures/Fig1.tif'
    fig.savefig(path,dpi=600,format='tiff',facecolor='white')
    with Image.open(path) as im:rgb=im.convert('RGB')
    rgb.save(path,compression='tiff_lzw',dpi=(600,600))
    plt.close(fig)
    p=out/'FIGURE_MANIFEST.csv';manifest=pd.read_csv(p)
    with Image.open(path) as im:
        assert im.mode=='RGB' and im.info['compression']=='tiff_lzw'
        for key,value in {'width_pixels':im.width,'height_pixels':im.height,'DPI':600,'RGB':True,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}.items():
            manifest.loc[manifest.file.eq('Fig1.tif'),key]=value
    manifest.to_csv(p,index=False)
    print('Updated Fig1.pdf, Fig1.png, Fig1.tif and figure manifest.')
if __name__=='__main__':main()
