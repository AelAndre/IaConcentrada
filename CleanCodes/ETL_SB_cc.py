#////////////////////Seoul Bikes ETL/////////////////////////////
#  Evan André Santana Pacheco A01769493
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
#////////////////////////////////////////////////////////////////

# This code purely focus on the ETL of Seoul bike data 
# Not the logic and procedure.

# ///// Libraries /////
from pathlib import Path
 
import pandas as pd
import numpy as np
 
# ///// Paths /////
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
 
# ///// Load Data  /////
df = pd.read_csv(DATA_DIR / 'SeoulBikeData.csv', encoding='latin1')
 
# ///// Transform Holiday /////
def changehol(text):
    if text == 'Holiday':
        return 1
    elif text == 'No Holiday':
        return 0
 
df['Holiday'] = df['Holiday'].apply(changehol)
 
# ////// Transform Seasons /////
 
def changesea(text):
    if text == 'Winter':
        return 1
    if text == 'Spring':
        return 2
    if text == 'Summer':
        return 3
    elif text == 'Autumn':
        return 4
 
df['Seasons'] = df['Seasons'].apply(changesea)
 
# ////// Transformar Functioning Day /////
 
def changefun(text):
    if text == 'Yes':
        return 1
    elif text == 'No':
        return 0
 
df['Functioning Day'] = df['Functioning Day'].apply(changefun)
 
# ///// Translate Visibility /////
df['Visibility (10m)'] = df['Visibility (10m)'].astype(float) / 2000.0
 
 
 
# ////// Imputation for non functioning days /////
 
def imput(df):
 
    df_im = df.copy()
    # cast to float so the interpolated means can be written without a dtype clash
    df_im['Rented Bike Count'] = df_im['Rented Bike Count'].astype(float)
    index_fund = df_im[df_im['Functioning Day'] == 0].index
 
    for idx in index_fund:
 
        before = 24
        while (idx - before >= 0) and (df_im.loc[idx - before, 'Functioning Day'] == 0):
            before += 24
 
        next_step = 24
        while (idx + next_step < len(df_im)) and (df_im.loc[idx + next_step, 'Functioning Day'] == 0):
            next_step += 24
 
        b_val = df_im.loc[idx - before, 'Rented Bike Count'] if (idx - before >= 0) else np.nan
        n_val = df_im.loc[idx + next_step, 'Rented Bike Count'] if (idx + next_step < len(df_im)) else np.nan
 
        # Interpolate
        if pd.notna(b_val) and pd.notna(n_val):
            imputation = (b_val + n_val) / 2
        elif pd.notna(b_val): 
            imputation = b_val
        elif pd.notna(n_val): 
            imputation = n_val
        else:
            imputation = 0
 
        df_im.loc[idx, 'Rented Bike Count'] = imputation
 
    return df_im
 
# ///// Imputation /////
df = imput(df)
 
 
# ///// Return all column values of Rented Bikes as Int /////
df['Rented Bike Count'] = df['Rented Bike Count'].round().astype(int)
 
# ///// Get weekends trhough Date /////
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Weekend'] = np.where(df['Date'].dt.dayofweek >= 5, 1, 0)
 
 
# ///// Divide by classes /////
# Per season, the last 216 hours (9 days) are separated from training.
# The first 96 of them (4 days) are assigned to validation and the last 120 (5 days)
# are assigned to test.
 
df['instance_id'] = df.index
 
TEST_HOURS_PER_SEASON = 120
VAL_HOURS_PER_SEASON = 96
 
season_change = df['Seasons'] != df['Seasons'].shift()
 
block_starts = df.index[season_change].tolist()
 
block_ends = block_starts[1:] + [len(df)]
 
train_parts, val_parts, test_parts, map_parts = [], [], [], []
 
for start, end in zip(block_starts, block_ends):
    block = df.iloc[start:end]
    holdout = TEST_HOURS_PER_SEASON + VAL_HOURS_PER_SEASON
 
    block_train = block.iloc[:-holdout]
    block_val = block.iloc[-holdout:-TEST_HOURS_PER_SEASON]
    block_test = block.iloc[-TEST_HOURS_PER_SEASON:]
 
    train_parts.append(block_train)
    val_parts.append(block_val)
    test_parts.append(block_test)
 
    map_parts.append(block_train.assign(Subset='train', local_id=range(1, len(block_train) + 1)))
    map_parts.append(block_val.assign(Subset='val', local_id=range(1, len(block_val) + 1)))
    map_parts.append(block_test.assign(Subset='test', local_id=range(1, len(block_test) + 1)))
 
df_train = pd.concat(train_parts)
df_val = pd.concat(val_parts)
df_test = pd.concat(test_parts)
 
map_df = pd.concat(map_parts).sort_values('instance_id')
map_df = map_df[['local_id', 'Date', 'Seasons', 'Subset', 'Functioning Day']].reset_index(drop=True)
 
drop_cols = ['Rented Bike Count', 'Date', 'Functioning Day', 'instance_id']
features = [col for col in df.columns if col not in drop_cols]
 
X_train, Y_train = df_train[features], df_train['Rented Bike Count']
X_val, Y_val = df_val[features], df_val['Rented Bike Count']
X_test, Y_test = df_test[features], df_test['Rented Bike Count']
 
 
# ///// Double Check /////
print(f"train: {len(X_train)} rows | val: {len(X_val)} rows | test: {len(X_test)} rows")
print(f"features ({len(features)}): {features}")
print("Null values:", int(df[features + ['Rented Bike Count']].isnull().sum().sum()))
 
# ///// Export /////
X_train.to_csv(DATA_DIR / 'X_train_seoul.csv', index=False)
Y_train.to_csv(DATA_DIR / 'Y_train_seoul.csv', index=False)
X_val.to_csv(DATA_DIR / 'X_val_seoul.csv', index=False)
Y_val.to_csv(DATA_DIR / 'Y_val_seoul.csv', index=False)
X_test.to_csv(DATA_DIR / 'X_test_seoul.csv', index=False)
Y_test.to_csv(DATA_DIR / 'Y_test_seoul.csv', index=False)
map_df.to_csv(DATA_DIR / 'map_seoulData.csv', index=False)
 
print("Exported: X_/Y_ train, val, test and map_seoulData.csv")