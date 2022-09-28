# Data
import pandas as pd
import argparse
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

wandb.init(project='Elec3')

parser = argparse.ArgumentParser()

parser.add_argument('--num_leaves', type=int, default=15)
parser.add_argument('--objective', type=str, default='regression')
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--min_data_in_leaf', type=int, default=4)
parser.add_argument('--max_depth', type=int, default=4)

args = parser.parse_args()

sunshine = pd.read_csv("Dataset/sunshine.csv")
temp = pd.read_csv("Dataset/temp.csv")
wind = pd.read_csv("Dataset/wind.csv")

train_data = sunshine.merge(temp, on=['Day', 'Hour'], how='left')
train_data = train_data.merge(wind, on=['Day', 'Hour'], how='left')

Temp_day = train_data.groupby(['Day'])['Temp'].mean().to_frame('Temp_day').reset_index()
Temp_day['Diff_Temp'] = Temp_day['Temp_day'].diff()
train_data = train_data.merge(Temp_day, on=['Day'], how='left')

features = [f for f in train_data.columns if f not in ['Radiation', 'day_t']]

params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': args.objective,
    'num_leaves': args.num_leaves, 
    'learning_rate': args.learning_rate,
    'metric': {'mse'},
    'verbose': -1,
    'min_data_in_leaf': args.min_data_in_leaf,
    'max_depth':args.max_depth,
    'seed': 42
}

lgb_train = lgb.Dataset(train_data[features], train_data['Radiation'].values)
# train
gbm = lgb.cv(params,
             train_set=lgb_train,
             stratified=False, 
             callbacks=[lgb.early_stopping(stopping_rounds=30)])


wandb.log({"mse": np.mean(gbm['l2-mean'])})
wandb.log({"stdv": np.mean(gbm['l2-stdv'])})