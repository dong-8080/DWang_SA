
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd
import numpy as np

df_merged_scores = pd.read_csv("./mci_scores.csv")

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1,
    'axes.linewidth': 0.5,
})


fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6))

columns = ['abeta', 'ptau', 'tau', 'fdg', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM', 'ADNI_VS']
titles = ['CSF-Aβ', 'CSF-pTau', 'CSF-Tau', 'FDG', 'ADNI-EF', 'ADNI-LAN', 'ADNI-MEM', 'ADNI-VS']
for i, col_name in enumerate(columns):
    row, col = i//4, i%4
    ax = axs[row, col]
    ax = pt.RainCloud(data=df_merged_scores, x="subtype", y=col_name, width_viol = 1, palette = palette, move=0, bw=0.2,  ax=ax, linewidth=0.5,
                      box_whiskerprops = {'linewidth': 0.5, "zorder": 2}, box_fliersize=1, box_linewidth=0.5,
                      box_medianprops = {"linewidth": 1, "color": "black"},
                      point_size=1)
    annotator = Annotator(pairs=pairs, data=df_merged_scores, x="subtype", y=col_name, order=[1, 2, 3, 4], ax=ax )
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', hide_non_significant=True, fontsize=9, 
                        line_offset=-1, text_offset=-5, line_width=0.5, line_height=0, verbose=False,
                        pvalue_thresholds=[[1e-4, "***"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]])
    annotator.apply_and_annotate()


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    ax.tick_params(left=False, axis="y", pad=-1)
    ax.tick_params(bottom=False, axis="x", pad=-1)
    
    ax.set_xticklabels(["ST1", "ST2", "ST3", "ST4"])
    ax.set_title(titles[i])
    
plt.tight_layout()