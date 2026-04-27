import pandas as pd
import numpy as np
from datetime import datetime
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_adni_subtype
from lifelines import KaplanMeierFitter
import matplotlib.gridspec as gridspec



palette = get_paletpalette = ['#5679BA', '#7F3F98', '#FFA45c', '#EE553D']
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


adni_info = pd.read_csv(r"./ADNI_MERGE.csv")
metadata_mci = pd.read_csv("./metadata_mci.csv")

sub_name = metadata_mci['subname']
mci_rids = [int(i.split("_")[-1]) for i in sub_name]

subtype = get_adni_subtype()
metadata_mci['subtype'] = subtype
np.random.shuffle(subtype)
metadata_mci['random_subtype'] = subtype
metadata_mci.drop(columns="confidence", inplace=True)

adni_info_filted = adni_info[adni_info.RID.isin(mci_rids)]

adni_info_filted.sort_values(by="RID", inplace=True)
adni_info_filted.sort_values(by="VISCODE", inplace=True)
adni_info_filted.dropna(subset="DX", inplace=True)

ptid_counts = adni_info_filted.groupby("PTID")['PTID'].count()
single_occurrence_ptids = ptid_counts[ptid_counts == 1].index

adni_info_filted = adni_info_filted[~adni_info_filted['PTID'].isin(single_occurrence_ptids)]

def convert_viscode_to_month(viscode):
    if viscode == 'bl':
        return 0
    elif viscode.startswith('m'):
        return int(viscode[1:])
    else:
        return None

from datetime import datetime
def calculate_date_difference(date_str1, date_str2):
    date_format = "%Y-%m-%d"
    
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)
    
    date_diff = date1 - date2
    
    return date_diff.days


def find_conversion_time(group):
    group = group.sort_values(by='time')
    conversion_time = None
    conversion_event = 0
    
    for i in range(1, len(group)):
        if group.iloc[i]['DX'] == 'Dementia' and group.iloc[i-1]['DX'] == 'MCI':
            conversion_time = group.iloc[i]['time']
            conversion_event = 1
            break
    
    if conversion_time is None:
        conversion_time = group['time'].max()
    
    return pd.Series({'time': conversion_time, 'event': conversion_event})

def find_conversion_time(group):
    group['EXAMDATE'] = pd.to_datetime(group['EXAMDATE'], errors='coerce')
    group['EXAMDATE'] = group['EXAMDATE'].dt.strftime('%Y-%m-%d')
    group = group.sort_values(by='EXAMDATE')
    conversion_time = None
    conversion_event = 0
    bl_examdate = group.iloc[0]['EXAMDATE']
    
    for i in range(1, len(group)):
        if group.iloc[i]['DX'] == 'Dementia' and group.iloc[i-1]['DX'] == 'MCI':
            conversion_time = calculate_date_difference(group.iloc[i]['EXAMDATE'], bl_examdate)
            conversion_event = 1
            break
    
    if conversion_time is None:
        conversion_time = calculate_date_difference(group['EXAMDATE'].max(), bl_examdate)
    
    return pd.Series({'time': conversion_time, 'event': conversion_event, 'bl': bl_examdate, "etime": group.iloc[i]['EXAMDATE']})


df = adni_info_filted[adni_info_filted['Month']<=60]
df['time'] = df['VISCODE'].apply(convert_viscode_to_month)
survival_data = df.groupby('PTID').apply(find_conversion_time).reset_index()
survival_data = pd.merge(survival_data, metadata_mci.loc[:, ["subname", "subtype", "random_subtype"]], left_on="PTID", right_on="subname").drop(columns="subname")
plt.figure(figsize=(3.6, 3))  

fitters = []
for subtype in [0,1,2,3]:
    mask = survival_data['subtype'] == subtype
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_data[mask]['time'], event_observed=survival_data[mask]['event'], label=f"ST{subtype+1}")
    fitters.append(kmf)
    kmf.plot_survival_function(ci_show=False, color=palette[subtype])

x_ticks_years = np.array([0, 1, 2, 3, 4, 5])
x_ticks_days = x_ticks_years * 365

plt.xticks(x_ticks_years*365, x_ticks_years)
plt.xlim(-120, 5.4*365)

add_at_risk_counts(*fitters, xticks=x_ticks_years*365)
plt.show()

# plot again

fig = plt.figure(figsize=(3.6, 4))
gs = gridspec.GridSpec(3, 1) 

ax1 = fig.add_subplot(gs[:2, 0])

for subtype in [0,1,2,3]:
    mask = survival_data['subtype'] == subtype
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_data[mask]['time'], event_observed=survival_data[mask]['event'], label=f"ST{subtype+1}")
    kmf.plot_survival_function(ci_show=False, color=palette[subtype])

x_ticks_years = np.array([0, 1, 2, 3, 4, 5])
x_ticks_days = x_ticks_years * 365

ax1.legend(fontsize=9)
ax1.set_xticks(x_ticks_years*365, x_ticks_years*12)
ax1.set_xlim(-0.5*365, 5.5*365)
ax1.set_xlabel("Years")
ax1.set_ylabel("Survival rates")
ax1.set_title("Progression of MCI convert to AD")

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(left=False, axis="y", pad=-1)
ax1.tick_params(bottom=False, axis="x", pad=-1)


ax2 = fig.add_subplot(gs[2, 0])
df = pd.DataFrame({
    '0': [111, 185, 319, 112],
    '1': [93, 164, 275, 87],
    '2': [69, 135, 205, 50],
    '3': [54, 103, 155, 33],
    '4': [31, 67, 100, 14],
    '5': [12, 21, 37, 6]
})

ax2 = sns.heatmap(df, annot=True, cmap=['white'], fmt='d', linecolor='white', cbar=False)

ax2.spines['bottom'].set_visible(True) 
ax2.spines['left'].set_visible(True)
ax2.set_yticklabels(['ST1', 'ST2', 'ST3', 'ST4'], rotation=0)
ax2.tick_params(left=False, axis="y", pad=-1)
ax2.tick_params(bottom=False, axis="x", pad=-1)
ax2.set_xlim(0, 6)
ax2.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax2.set_xticklabels([0, 12, 24, 36, 48, 60])
ax2.set_ylabel("At risk")
ax2.set_xlabel("Months")

plt.savefig(r"km.png", dpi=1200, bbox_inches="tight")
plt.show()