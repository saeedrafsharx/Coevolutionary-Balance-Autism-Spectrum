"""Vector heatmap cells, avoiding interpolation artifacts in exported PDFs."""
import numpy as np

def heatmap_cells(ax,values,*,cmap='viridis',vmin=None,vmax=None,aspect='auto'):
    values=np.asarray(values,float)
    rows,columns=values.shape
    artist=ax.pcolormesh(np.arange(columns+1)-.5,np.arange(rows+1)-.5,
                        np.ma.masked_invalid(values),cmap=cmap,vmin=vmin,vmax=vmax,
                        shading='flat',edgecolors='face',linewidth=0)
    ax.set_xlim(-.5,columns-.5);ax.set_ylim(rows-.5,-.5)
    ax.set_aspect(aspect);ax.set_facecolor('#d9d9d9')
    return artist
