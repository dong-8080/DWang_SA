from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

FEATURE_DIR = r"D:\Radiomics_BN246_subset"
FILE_PATTERN = "*.csv"

OUT_DIR = r"D:\normalized"
os.makedirs(OUT_DIR, exist_ok=True)

MINMAX_SAVE_DIR = r"D:\normalized\minmax"
ZSCORE_SAVE_DIR = r"D:\normalized\zscored"
ROBUST_SAVE_DIR = r"D:\normalized\robust"
os.makedirs(MINMAX_SAVE_DIR, exist_ok=True)
os.makedirs(ZSCORE_SAVE_DIR, exist_ok=True)
os.makedirs(ROBUST_SAVE_DIR, exist_ok=True)
SAVE_DIR_MAP = {
    "minmax_col": MINMAX_SAVE_DIR,
    "zscore_col": ZSCORE_SAVE_DIR,
    "robust_col": ROBUST_SAVE_DIR,
}

FEATURE_MASK = np.array([
    1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0,
    0, 1, 1, 0, 0, 0, 1
], dtype=bool)
BASELINE_NORM = "minmax_col"
COMPARE_NORMS = ["zscore_col", "robust_col"]
CALCULATE_EDGE_OVERLAP = True
TOP_EDGE_RATIO = 0.10

def minmax_col(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    col_min = np.min(feat, axis=0, keepdims=True)
    col_max = np.max(feat, axis=0, keepdims=True)
    denom = col_max - col_min
    denom[denom < eps] = 1.0
    out = (feat - col_min) / denom
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def zscore_col(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    col_mean = np.mean(feat, axis=0, keepdims=True)
    col_std = np.std(feat, axis=0, keepdims=True)
    col_std[col_std < eps] = 1.0
    out = (feat - col_mean) / col_std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
def robust_col(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    col_median = np.median(feat, axis=0, keepdims=True)
    q1 = np.percentile(feat, 25, axis=0, keepdims=True)
    q3 = np.percentile(feat, 75, axis=0, keepdims=True)
    iqr = q3 - q1
    iqr[iqr < eps] = 1.0
    out = (feat - col_median) / iqr
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
def apply_normalization(feat: np.ndarray, method: str) -> np.ndarray:
    if method == "minmax_col":
        return minmax_col(feat)
    elif method == "zscore_col":
        return zscore_col(feat)
    elif method == "robust_col":
        return robust_col(feat)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def build_r2sn(feat: np.ndarray, norm_method: str) -> np.ndarray:
    feat = np.asarray(feat, dtype=float)
    feat_norm = apply_normalization(feat, norm_method)

    net = np.corrcoef(feat_norm)
    net = np.nan_to_num(net, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(net, 1.0)

    return net

def upper_triangle(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices_from(mat, k=1)]

def network_similarity(net1: np.ndarray, net2: np.ndarray) -> float:
    v1 = upper_triangle(net1)
    v2 = upper_triangle(net2)

    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan

    return float(spearmanr(v1, v2).correlation)

def strongest_edge_mask(net: np.ndarray, top_ratio: float = 0.10) -> np.ndarray:
    v = upper_triangle(net)
    n_edges = len(v)
    k = max(1, int(np.floor(n_edges * top_ratio)))

    abs_v = np.abs(v)
    threshold = np.partition(abs_v, -k)[-k]
    mask = abs_v >= threshold
    return mask

def jaccard_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return np.nan
    return inter / union


def plot_similarity_boxplot(results_df: pd.DataFrame, out_path: str) -> None:
    pairs = results_df["comparison"].unique().tolist()
    data = [
        results_df.loc[results_df["comparison"] == pair, "similarity"].dropna().values
        for pair in pairs
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=pairs, showfliers=True)
    plt.ylabel("Spearman similarity")
    plt.title("Normalization robustness of R2SN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_overlap_boxplot(results_df: pd.DataFrame, out_path: str) -> None:
    pairs = results_df["comparison"].unique().tolist()
    data = [
        results_df.loc[results_df["comparison"] == pair, "edge_overlap"].dropna().values
        for pair in pairs
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=pairs, showfliers=True)
    plt.ylabel("Jaccard overlap of strongest edges")
    plt.title("Strongest-edge overlap across normalization methods")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

files = sorted(glob.glob(os.path.join(FEATURE_DIR, FILE_PATTERN)))
print("Number of subjects:", len(files))

all_results = []
for file in files:
    subject_id = os.path.splitext(os.path.basename(file))[0]

    df = pd.read_csv(file)
    feat = df.values.astype(float)

    feat_selected = feat[:, FEATURE_MASK]

    net_base = build_r2sn(feat_selected, BASELINE_NORM)
    np.save(os.path.join(SAVE_DIR_MAP[BASELINE_NORM], f"{subject_id}.npy"), net_base)

    for norm_method in COMPARE_NORMS:
        net_cmp = build_r2sn(feat_selected, norm_method)
        np.save(os.path.join(SAVE_DIR_MAP[norm_method], f"{subject_id}.npy"), net_cmp)

        sim = network_similarity(net_base, net_cmp)

        result = {
            "subject_id": subject_id,
            "baseline_norm": BASELINE_NORM,
            "compare_norm": norm_method,
            "comparison": f"{BASELINE_NORM}_vs_{norm_method}",
            "similarity": sim
        }

        if CALCULATE_EDGE_OVERLAP:
            mask_base = strongest_edge_mask(net_base, TOP_EDGE_RATIO)
            mask_cmp = strongest_edge_mask(net_cmp, TOP_EDGE_RATIO)
            overlap = jaccard_overlap(mask_base, mask_cmp)
            result["edge_overlap"] = overlap

        all_results.append(result)

results_df = pd.DataFrame(all_results)

results_path = os.path.join(OUT_DIR, "subject_level_similarity_spearmanr.csv")
results_df.to_csv(results_path, index=False)

summary_df = (
    results_df
    .groupby("comparison")
    .agg(
        mean_similarity=("similarity", "mean"),
        sd_similarity=("similarity", "std"),
        median_similarity=("similarity", "median"),
        min_similarity=("similarity", "min"),
        max_similarity=("similarity", "max"),
        q1_similarity=("similarity", lambda x: x.quantile(0.25)),
        q3_similarity=("similarity", lambda x: x.quantile(0.75))
    )
    .reset_index()
)

if CALCULATE_EDGE_OVERLAP:
    overlap_summary = (
        results_df
        .groupby("comparison")
        .agg(
            mean_edge_overlap=("edge_overlap", "mean"),
            sd_edge_overlap=("edge_overlap", "std"),
            median_edge_overlap=("edge_overlap", "median"),
            min_edge_overlap=("edge_overlap", "min"),
            max_edge_overlap=("edge_overlap", "max"),
            q1_edge_overlap=("edge_overlap", lambda x: x.quantile(0.25)),
            q3_edge_overlap=("edge_overlap", lambda x: x.quantile(0.75))
        )
        .reset_index()
    )
    summary_df = summary_df.merge(overlap_summary, on="comparison", how="left")

summary_path = os.path.join(OUT_DIR, "group_level_summary_spearmanr.csv")
summary_df.to_csv(summary_path, index=False)

plot_similarity_boxplot(
    results_df,
    os.path.join(OUT_DIR, "normalization_similarity_boxplot_spearmanr.png")
)

if CALCULATE_EDGE_OVERLAP:
    plot_overlap_boxplot(
        results_df,
        os.path.join(OUT_DIR, "normalization_edge_overlap_boxplot_spearmanr.png")
    )

# for network stability anslysis
FEATURE_DIR = r"D:\Radiomics_BN246_subset"
FILE_PATTERN = "*.csv"

SUBSET_SIZE = 20
N_REPEAT = 100
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_MASK = np.array([
    1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0,
    0, 1, 1, 0, 0, 0, 1
], dtype=bool)

def build_r2sn(feat):
    feat = np.asarray(feat, dtype=float)
    row_min = feat.min(axis=0, keepdims=True)
    row_max = feat.max(axis=0, keepdims=True)
    denom = row_max - row_min
    denom[denom == 0] = 1.0
    feat_norm = (feat - row_min) / denom
    net = np.corrcoef(feat_norm)
    net = np.nan_to_num(net, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(net, 1.0)
    return net

def upper_triangle(mat):
    return mat[np.triu_indices_from(mat, k=1)]

def network_similarity(net1, net2):
    v1 = upper_triangle(net1)
    v2 = upper_triangle(net2)

    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan
    return np.corrcoef(v1, v2)[0, 1]

files = sorted(glob.glob(os.path.join(FEATURE_DIR, FILE_PATTERN)))
print("Number of subjects:", len(files))
all_similarity = []

for file in files:
    subject_id = os.path.basename(file)

    df = pd.read_csv(file)
    feat = df.values.astype(float)
    feat_selected = feat[:, FEATURE_MASK]
    n_selected_features = feat_selected.shape[1]
    baseline_net = build_r2sn(feat_selected)
    feature_idx = np.arange(n_selected_features)

    subject_similarity = []

    for i in range(N_REPEAT):
        idx = np.random.choice(feature_idx, SUBSET_SIZE, replace=False)
        feat_sub = feat_selected[:, idx]
        net_sub = build_r2sn(feat_sub)
        sim = network_similarity(baseline_net, net_sub)
        subject_similarity.append(sim)
        
    subject_similarity = np.array(subject_similarity)
    all_similarity.append(subject_similarity)
    print(
        f"{subject_id} | selected_features={n_selected_features} | "
        f"mean={np.nanmean(subject_similarity):.4f}, "
        f"std={np.nanstd(subject_similarity):.4f}"
    )


all_similarity = np.array(all_similarity)
np.save(r"D:\workspace\gradient\revised\R2SN\feature_similarity.npy", all_similarity)

# for vusalization
df_compared = pd.read_csv(r"D:\workspace\gradient\revised\R2SN\normalized\subject_level_similarity.csv")
df_compared_pearson = pd.read_csv(r"D:\workspace\gradient\revised\R2SN\normalized\subject_level_similarity.csv")
df_compared_pearson = df_compared_pearson[df_compared_pearson['compare_norm']=="zscore_col"]
df_compared_spearman = pd.read_csv(r"D:\workspace\gradient\revised\R2SN\normalized\subject_level_similarity_spearmanr.csv")
df_compared_spearman = df_compared_spearman[df_compared_spearman['compare_norm']=="zscore_col"]


data1 = df_compared_pearson['similarity']
data2 = df_compared_spearman['similarity']
df_plot = pd.DataFrame({
    'similarity': np.concatenate([data1, data2]),
    'comparison': ['minmax_col_vs_zscore_col'] * len(data1) + ['minmax_col_vs_robust_col'] * len(data2)
})
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
sns.set_theme(style="ticks", rc={
    "axes.spines.right": True,
    "axes.spines.top": True,
    "font.family": "sans-serif",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})
plt.figure(figsize=(5, 4), dpi=300)
palette = ["#7FB3D5", "#F7DC6F"]
ax = sns.boxplot(
    data=df_plot,
    x="comparison",
    y="similarity",
    palette=palette,
    width=0.6,
    linewidth=1.0,       
    showfliers=True,      
    fliersize=3,           
    flierprops=dict(marker='o', markersize=3, color='black', alpha=0.5)
)
new_labels = ['Pearson', 'Spearman']
ax.set_xticklabels(new_labels)
plt.ylabel("Similarity Score", fontsize=12)
plt.xlabel("Correlation metric", fontsize=12)
plt.ylim(0.18, 0.92)
ax.grid(False)
sns.despine(top=False, right=False)
plt.tight_layout()
plt.show()

similarity = np.load(r"D:\R2SN\feature_similarity.npy")
df_similarity = pd.DataFrame(similarity)
mean_similarity = df_similarity.mean(axis=1)
std_similarity = df_similarity.std(axis=1)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
sns.set_theme(style="ticks")
np.random.seed(42)
mean_data = mean_similarity
var_data = std_similarity
fig, ax1 = plt.subplots(figsize=(6, 5), dpi=150)
sns.histplot(
    mean_data, bins=65, kde=True, 
    color="#748CB1", edgecolor='white', linewidth=0.5,
    line_kws={"color": "#1F3A5F", "lw": 1.5}, 
    ax=ax1
)
ax1.set_xlabel("Mean Similarity Coefficient", fontsize=11, labelpad=8)
ax1.set_ylabel("Frequency (Count)", fontsize=11, labelpad=8)
ax1.set_xlim([0.968, 0.991])
ax1.set_xticks(
    [0.970, 0.974, 0.978, 0.982, 0.986, 0.99], 
    ["0.970", "0.974", "0.978", "0.982", "0.986", "0.990"])

ax1.tick_params(direction='in', width=1.0, labelsize=9)
for spine in ax1.spines.values():
    spine.set_linewidth(1.0) 

ax_inset = inset_axes(ax1, width="35%", height="30%", loc=2, borderpad=3)
sns.histplot(var_data, bins=30, color="#B0BCCB", edgecolor='none', ax=ax_inset)
ax_inset.set_title("Variance Distribution", fontsize=8, fontweight='bold', pad=4)
ax_inset.set_xlabel("Variance", fontsize=7)
ax_inset.set_ylabel("", fontsize=0)

for spine in ax_inset.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color('#333333')
ax_inset.tick_params(
    axis='both', 
    which='major', 
    labelsize=6, 
    direction='in', 
    width=0.6,
    length=2
)
ax_inset.xaxis.set_major_locator(plt.MaxNLocator(3))
ax_inset.yaxis.set_major_locator(plt.MaxNLocator(3))

plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.15)

plt.savefig(r'D:\workspace\gradient\revised\plot\Similarity_Distribution_Science_Style.png', bbox_inches='tight', dpi=300)
plt.show()