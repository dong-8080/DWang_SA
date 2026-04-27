import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import os


pacc_path = r"D:\PACC.csv"
df_pacc = pd.read_csv(pacc_path)
df_pacc = df_pacc[df_pacc['SUBSTUDY']=="A4"]
df_pacc = df_pacc[df_pacc['VISCODE']<200]

df_clinical_drug = pd.read_csv(r"D:\df_clinical_drug.csv")
df_clinical_placebo = pd.read_csv(r"D:\df_clinical_placebo.csv")
df_a4_all = pd.concat([df_clinical_drug, df_clinical_placebo])
matrix_H_A4 = pd.read_csv(r"D:\NMF_H_a4.csv")
matrix_W_A4 = pd.read_csv(r"D:\NMF_W_a4.csv")

A4_subtype = matrix_H_A4.idxmax().to_list()
df_clinical_drug['subtype']  = A4_subtype

def get_adni_subtype():
    nmf_path = r"D:\NMF"
    matrix_H_adni = pd.read_csv(os.path.join(nmf_path, "NMF_H_adni.csv"))
    return matrix_H_adni.idxmax().to_list()

datapath = r"D:\adni_gradient_mci.csv"
gradient_adni_mci = pd.read_csv(datapath, header=None)
gradient_adni_mci['subtype'] = get_adni_subtype()

gradient_adni_mci_template = gradient_adni_mci.groupby("subtype").mean().reset_index(drop=True)
gradient_adni_mci_template


def calculate_distances(gradient_adni_ad, gradient_adni_mci_template):
    rows, _ = gradient_adni_ad.shape
    num_vectors, _ = gradient_adni_mci_template.shape
    distance_matrix = np.zeros((rows, num_vectors))
    for i in range(rows):
        for j in range(num_vectors):
            distance_matrix[i, j] = np.linalg.norm(
                gradient_adni_ad.iloc[i].values - gradient_adni_mci_template.iloc[j].values)

    return pd.DataFrame(distance_matrix)

distance_matrix = calculate_distances(df_a4_all.iloc[:, 5:], gradient_adni_mci_template)
mapped_subtype = distance_matrix.idxmin(axis=1)
mapped_subtype.value_counts()
df_a4_all['subtype']  = mapped_subtype

df_pacc_merged = pd.merge(df_a4_all[['BID', 'subtype', 'TX', 'AGEYR', 'SEX']], df_pacc, on="BID", how='right')
df_pacc_merged.dropna(subset=['subtype'], inplace=True)

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
palette = ['#5679BA', '#7F3F98', '#FFA45c', '#EE553D']

df_pacc_merged = df_pacc_merged[df_pacc_merged['VISCODE']>=6]

df = df_pacc_merged.dropna(subset=['subtype', 'PACC']).copy()

df['PACC_std'] = (df['PACC'] - df['PACC'].mean())/df['PACC'].std()
df['VISCODE'] = df['VISCODE'].astype(int)
df['subtype'] = df['subtype'].astype('category')
df['SEX'] = df['SEX'].astype('category')
df['subtype'] = df['subtype'].cat.reorder_categories(
    [0.0, 1.0, 2.0, 3.0], ordered=False
)
df['TX'] = df['TX'].astype('category')
df_subgroups = [df[df['TX']=="Placebo"], df[df['TX']=="Solanezumab"]]
titles = ['Placebo', 'Solanezumab']
fig, axs = plt.subplots(1, 2, figsize=(1.8*2, 2))

for idx, df_subgroup in enumerate(df_subgroups):
    ax = axs[idx]
    df_subgroup.reset_index(inplace=True, drop=True)
    model = smf.mixedlm("PACC_std ~ VISCODE + subtype + VISCODE:subtype + AGEYR + SEX", df_subgroup, groups=df_subgroup['BID'], re_formula="1")
    result = model.fit()
    print(result.summary())
    df_subgroup['fitted'] = result.fittedvalues
    ax = axs[idx]
    for i in range(4): 
        df_plot = df_subgroup[df_subgroup['subtype']==i]
        sns.regplot(data=df_plot, x='VISCODE', y='fitted', scatter_kws={'s': 1},scatter=False, color=palette[i], ax=ax)
    ax.set_title(titles[idx])
    if idx==0:
        ax.set_ylabel("z-scored ΔPACC")
    else:
        ax.set_ylabel("")
    ax.set_ylim([-2, 0.5])
    ax.set_xlabel("Months")
    ax.set_xticks([0, 18, 36, 54, 72, 90, 108])
    ax.tick_params(left=False, axis='y', pad=-1)    
    ax.tick_params(bottom=False, axis='x', pad=-1)
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 

plt.tight_layout()
plt.show()


df = df_pacc_merged.dropna(subset=['subtype', 'PACC']).copy()
df['PACC_std'] = (df['PACC'] - df['PACC'].mean())/df['PACC'].std()
fig, axs = plt.subplots(1, 4, figsize=(1.8*4, 2))

for i in range(4):    
    ax = axs[i]
    df_subgroup = df[df['subtype']==i].copy()
    df_subgroup.reset_index(inplace=True, drop=True)
    df_subgroup['VISCODE'] = df_subgroup['VISCODE'].astype(int)
    df = df[df['VISCODE']!=0]
    df_subgroup['subtype'] = df_subgroup['subtype'].astype('category')

    df_subgroup['AGEYR'] = (df_subgroup['AGEYR'] - df_subgroup['AGEYR'].min()) / (df_subgroup['AGEYR'].max()-df_subgroup['AGEYR'].min())

    df_subgroup['SEX'] = df_subgroup['SEX'].astype('category')
    model = smf.mixedlm("PACC_std ~ VISCODE + TX + VISCODE:TX + AGEYR + SEX", df_subgroup, groups=df_subgroup['BID'], re_formula="1")
    result = model.fit()
    df_subgroup['fitted'] = result.fittedvalues
    df_plot = df[df['subtype']==i]
    sns.regplot(data=df_subgroup[df_subgroup['TX']=="Placebo"], x='VISCODE', y='PACC_std', scatter_kws={'s': 1},scatter=False, color="#1d73b6", ax=ax)
    sns.regplot(data=df_subgroup[df_subgroup['TX']=="Solanezumab"], x='VISCODE', y='PACC_std', scatter_kws={'s': 1},scatter=False, color="#f27830", ax=ax)
    if i==0:
        ax.set_ylabel("z-scored ΔPACC")
    else:
        ax.set_ylabel("")
    ax.set_xlabel('Months')
    ax.set_xticks([0, 18, 36, 54, 72, 90, 108])
    if i<3:
        ax.set_yticks([-0.8, -0.4, 0, 0.4])
        ax.set_ylim([-0.9, 0.5])
    ax.set_title(f"ST{i+1}")
    ax.tick_params(left=False, axis='y', pad=-1)    
    ax.tick_params(bottom=False, axis='x', pad=-1)
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
plt.tight_layout()
plt.show()

df = df_pacc_merged.dropna(subset=['subtype', 'PACC']).copy()

df['PACC_std'] = (df['PACC'] - df['PACC'].mean())/df['PACC'].std()
df['TX'] = df['TX'].astype('category')
df["time"] = df["VISCODE"] - df["VISCODE"].mean()
df["BID"] = df["BID"].astype(str)
df["subtype"] = df["subtype"].astype("category")
df["SEX"] = df["SEX"].astype("category") 
model = smf.mixedlm(
    "PACC_std ~ VISCODE * TX * subtype + AGEYR + SEX",
    df,
    groups=df["BID"],
    re_formula="1"
)
result = model.fit(method="lbfgs")
print(result.summary())
wald_results = result.wald_test_terms()
wald_results.summary_frame()