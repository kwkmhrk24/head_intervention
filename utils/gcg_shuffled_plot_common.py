"""Shared plotting configuration for phase regional/additional figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-gcg-shuffled-regions')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finish(fig,path,title):
    fig.suptitle(title,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
