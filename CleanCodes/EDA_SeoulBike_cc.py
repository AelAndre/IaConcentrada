#//////////////////// Seoul Bikes EDA ////////////////////////////
#  Evan Andre Santana Pacheco A01769493
# TC3009C.601
#
# Predictive Modeling of Urban Bike-Sharing Demand in Seoul
#   Seoul Bike Sharing Demand (UCI Machine Learning Repository)
# Author(s)
#   ------
# https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
#
# Licence
#    Creative Commons Attribution 4.0 International (CC BY 4.0)
#
# References
#   Hue function
#   https://www.geeksforgeeks.org/python/seaborn-lineplot-method-in-python/
#   pd.cut for discrete bins
#   https://pandas.pydata.org/docs/reference/api/pandas.cut.html
#
#////////////////////////////////////////////////////////////////

# This code purely focus on the EDA of Seoul bike data
# Not the logic and procedure.
# Figures are written to figures/ and LaTeX tables to tables/.

# ///// Libraries /////
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ///// Paths /////
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
FIGURE_DIR = PROJECT_DIR / 'Figures'
TABLE_DIR = PROJECT_DIR / 'Tables'
 
FIGURE_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def save_figure(name):
    plt.savefig(FIGURE_DIR / name, dpi=150, bbox_inches='tight')
    plt.close()


def save_table(table, name, **kwargs):
    kwargs.setdefault('escape', True)
    kwargs.setdefault('float_format', '%.2f')
    (TABLE_DIR / name).write_text(table.to_latex(**kwargs), encoding='utf-8')


# ///// Load Data /////
Y_train = pd.read_csv(DATA_DIR / 'Y_train_seoul.csv', encoding='utf-8')
X_train = pd.read_csv(DATA_DIR / 'X_train_seoul.csv', encoding='utf-8')
Y_val = pd.read_csv(DATA_DIR / 'Y_val_seoul.csv', encoding='utf-8')
X_val = pd.read_csv(DATA_DIR / 'X_val_seoul.csv', encoding='utf-8')
Y_test = pd.read_csv(DATA_DIR / 'Y_test_seoul.csv', encoding='utf-8')
X_test = pd.read_csv(DATA_DIR / 'X_test_seoul.csv', encoding='utf-8')

map_df = pd.read_csv(DATA_DIR / 'map_seoulData.csv', encoding='latin1')
map_df['Date'] = pd.to_datetime(map_df['Date'])

eda = pd.concat([Y_train, X_train], axis=1)


# ///// Variable groups /////
tempcate = ['Seasons', 'Holiday', 'Weekend']
categorical = eda[tempcate]
discrete = eda['Hour']

continous = eda.drop(tempcate, axis=1).drop('Hour', axis=1)
contcols = list(continous.columns)


# ///// Histograms /////
fig, axes = plt.subplots(4, 3, figsize=(18, 15))
axes = axes.flatten()

for i, col in enumerate(contcols):
    sns.histplot(continous[col], kde=True, ax=axes[i])
    axes[i].set_title(f'1.{i+1} Histogram of {col}', fontsize=14)

sns.histplot(categorical['Holiday'], ax=axes[9])
axes[9].set_title('1.10 Histogram of Holiday', fontsize=14)

for j in range(10, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
save_figure('fig1_histograms.png')


# ///// Boxplots of continous variables /////
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(contcols):
    sns.boxplot(x=eda[col], ax=axes[i])
    axes[i].set_title(f'2.{i+1} Boxplot of {col}', fontsize=14)
    axes[i].set_xlabel('')

for j in range(len(contcols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
save_figure('fig2_boxplots.png')


# ///// Boxplots of Rented Bike Count per categorical variable /////
plt.figure(figsize=(15, 5))

for i, col in enumerate(tempcate, 10):
    plt.subplot(1, 3, i - 9)
    sns.boxplot(data=eda, x=col, y='Rented Bike Count')
    plt.title(f'2.{i} Distribution per {col}', fontsize=14)
    plt.ylabel('Rented Bikes')

plt.tight_layout()
save_figure('fig3_boxplots_categorical.png')


# ///// Line plot of the created Weekend feature /////
plt.figure(figsize=(12, 6))
sns.lineplot(data=eda, x='Hour', y='Rented Bike Count', hue='Weekend', marker='o')
plt.xticks(range(0, 24))
plt.title('3.1 Daily Rented Bike demand in a day (Weekends vs Weekdays)')
plt.xlabel('Approximate hour of the day (0-23)')
plt.ylabel('Rented Bike count averaged')
plt.grid(True, linestyle='--')
save_figure('fig4_demand_weekend.png')


# ///// Hue classes for the remaining features /////
eda['Rain'] = (eda['Rainfall(mm)'] > 0).map({True: 'Rain', False: 'No raining'})
eda['Solar Radiation'] = (eda['Solar Radiation (MJ/m2)'] > 0).map({True: 'Solar Rad', False: 'No solar Rad'})
eda['Snow'] = (eda['Snowfall (cm)'] > 0).map({True: 'Snow', False: 'No snow'})

eda['Temperature'] = pd.cut(eda['Temperature(°C)'], bins=[-20, 0, 20, 40],
                            labels=['Low', 'Medium', 'High'], include_lowest=True)
eda['Humidity'] = pd.cut(eda['Humidity(%)'], bins=[0, 30, 70, 100],
                         labels=['Dry', 'Medium', 'Humid'], include_lowest=True)
eda['Wind speed'] = pd.cut(eda['Wind speed (m/s)'], bins=[0, 2, 4, 8],
                           labels=['Low', 'Medium', 'High'], include_lowest=True)
eda['Visibility percentage'] = pd.cut(eda['Visibility (10m)'], bins=[0, 0.33, 0.66, 1.0],
                                      labels=['Low', 'Medium', 'High'], include_lowest=True)

hue_variables = ['Seasons', 'Holiday', 'Rain', 'Snow', 'Solar Radiation',
                 'Temperature', 'Humidity', 'Wind speed', 'Visibility percentage']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, hue_var in enumerate(hue_variables):
    sns.lineplot(data=eda, x='Hour', y='Rented Bike Count', hue=hue_var, marker='o', ax=axes[i])
    axes[i].set_xticks(range(0, 24, 4))
    axes[i].set_title(f'3.{i+2} Demand per hour regarding: {hue_var}')
    axes[i].grid(True, linestyle='--')

for j in range(len(hue_variables), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
save_figure('fig5_demand_by_feature.png')


# ///// Subsets tagged for comparison /////
map_train = map_df[map_df['Subset'] == 'train'].reset_index(drop=True)
map_val = map_df[map_df['Subset'] == 'val'].reset_index(drop=True)
map_test = map_df[map_df['Subset'] == 'test'].reset_index(drop=True)

X_train_tag = X_train.copy(); X_train_tag['Subset'] = 'train'
X_val_tag = X_val.copy(); X_val_tag['Subset'] = 'val'
X_test_tag = X_test.copy(); X_test_tag['Subset'] = 'test'

X_train_tag['Rented Bike Count'] = Y_train.values
X_val_tag['Rented Bike Count'] = Y_val.values
X_test_tag['Rented Bike Count'] = Y_test.values

compare_df = pd.concat([X_train_tag, X_val_tag, X_test_tag], ignore_index=True)

eda_dated = eda.copy()
eda_dated['Date'] = map_train['Date']


# ///// Daily total of the train subset /////
full_range = pd.date_range(map_df['Date'].min(), map_df['Date'].max(), freq='D')

daily_trainsum = (eda_dated.set_index('Date')['Rented Bike Count']
                  .resample('D').sum(min_count=1).reindex(full_range))

plt.figure(figsize=(14, 5))
daily_trainsum.plot(marker='o', markersize=2)
plt.title('4.1 Daily total of rented bikes (train subset only)')
plt.ylabel('Daily total')
plt.grid(True, linestyle='--')
save_figure('fig6_daily_train.png')


# ///// Daily total across every subset /////
compare_dated = compare_df.copy()
compare_dated['Date'] = pd.concat(
    [map_train['Date'], map_val['Date'], map_test['Date']], ignore_index=True
)

daily_rb_subset = compare_dated.groupby(['Date', 'Subset'])['Rented Bike Count'].sum().reset_index()

plt.figure(figsize=(14, 5))
sns.scatterplot(data=daily_rb_subset, x='Date', y='Rented Bike Count', hue='Subset', s=20)
plt.title('4.2 Daily total rented bikes across the year (Train, Validation and Test subsets)')
plt.ylabel('Daily total')
save_figure('fig7_daily_all_subsets.png')


# ///// Descriptive tables /////
def outlierdata_visualizer(df, cols, group_cols, stats=['mean', 'std', 'min', 'max']):
    return df.groupby(group_cols)[cols].agg(stats).round(2)


def event_frequency(series):
    return round((series > 0).mean() * 100, 1)


weather_check_cols = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                      'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
                      'Rainfall(mm)', 'Snowfall (cm)']

critical_outlier = ['Temperature(°C)', 'Rainfall(mm)', 'Snowfall (cm)', 'Wind speed (m/s)']

table_descriptive = outlierdata_visualizer(compare_df, weather_check_cols, group_cols=['Subset']).T
save_table(table_descriptive, 'table4_descriptive.tex')

blocks = {}
for var in critical_outlier:
    blocks[var] = outlierdata_visualizer(compare_df, [var], group_cols=['Seasons', 'Subset'])[var]

table_critical = pd.concat(blocks, names=['Variable'])
save_table(table_critical, 'table5_critical.tex', longtable=True,
           caption='Critical statistical values for all subsets',
           label='tab:critical_outlier')

table_rain_snow = compare_df.groupby(['Seasons', 'Subset'])[['Rainfall(mm)', 'Snowfall (cm)']].agg(
    ['max', event_frequency])
table_rain_snow = table_rain_snow.rename(columns={'event_frequency': '% of hours'}, level=1)
save_table(table_rain_snow, 'table6_rain_snow.tex')

table_rented = outlierdata_visualizer(compare_df, ['Rented Bike Count'], group_cols=['Seasons', 'Subset'])
save_table(table_rented, 'table7_rented_bikes.tex')


# ///// Zoom out /////
plt.figure(figsize=(12, 6))
sns.boxplot(data=compare_df, x='Seasons', y='Rented Bike Count', hue='Subset')
plt.title('5.1 Distribution of Rented Bike Count per Season per Subset')
plt.grid(True, linestyle='--', alpha=0.6)
save_figure('fig8_zoomout_season.png')

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
sns.boxplot(data=compare_df, x='Subset', y='Temperature(°C)', ax=axes[0])
sns.boxplot(data=compare_df, x='Subset', y='Rented Bike Count', ax=axes[1])
sns.boxplot(data=compare_df, x='Subset', y='Wind speed (m/s)', ax=axes[2])

event_pct = compare_df.groupby('Subset')[['Rainfall(mm)', 'Snowfall (cm)']].agg(event_frequency).reset_index()
event_pct_melt = event_pct.melt(id_vars='Subset', var_name='Event', value_name='% of hours with event')
sns.barplot(data=event_pct_melt, x='Subset', y='% of hours with event', hue='Event', ax=axes[3])

for ax, title in zip(axes, ['5.2 Temperature', '5.3 Rented Bike Count',
                            '5.4 Wind speed', '5.5 Instances with rain and snow']):
    ax.set_title(title)

plt.tight_layout()
save_figure('fig9_zoomout_features.png')


# ///// Correlation matrix /////
original_cols = ['Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
                 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
                 'Rainfall(mm)', 'Snowfall (cm)', 'Seasons', 'Holiday', 'Weekend']

r = eda[original_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(r, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('6.1 Correlation Matrix of Seoul Bike Data')
save_figure('fig10_correlation.png')

save_table(r.round(2), 'table8_correlation.tex')


# ///// Output /////
print(f"Figures saved to: {FIGURE_DIR}")
print(f"LaTeX tables saved to: {TABLE_DIR}")