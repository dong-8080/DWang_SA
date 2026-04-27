from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

graident_mci = pd.read_csv(r"D:\gradient_prim_mci.csv", header=None)
df_metadata = pd.read_csv(r"D:\metadata_mci.csv")
df = pd.concat([df_metadata, graident_mci], axis=1)
X = df.iloc[:, -246:].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca_2d = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
groups = df['subtype'].values
unique_groups = np.unique(groups)
colors = ['#5679BA', '#7F3F98', '#FFA45c', '#EE553D']
legend_labels = ['ST1', 'ST2', 'ST3', 'ST4']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 11,
    'axes.linewidth': 1.0,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})
fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=300)

for i, g in enumerate(unique_groups):
    idx = groups == g
    ax.scatter(
        X_pca_2d[idx, 0],
        X_pca_2d[idx, 1],
        s=28,
        c=colors[i],
        label=legend_labels[i],
        alpha=0.85,
        edgecolors='white',
        linewidths=0.4
    )
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.tick_params(axis='both', which='both', width=1.0, length=4, direction='out')
ax.legend(
    frameon=False,
    loc='best',
    handletextpad=0.4,
    borderpad=0.2,
    labelspacing=0.4
)

plt.tight_layout()
plt.show()