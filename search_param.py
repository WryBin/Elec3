# Data
import pandas as pd
import argparse
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

wandb.init(project='Elec3')

parser = argparse.ArgumentParser()

parser.add_argument('--num_leaves', type=int, default=15)

# signal parameters
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--min_data_in_leaf', type=int, default=4)
parser.add_argument('--max_depth', type=int, default=4)

args = parser.parse_args()

sunshine = pd.read_csv("Dataset/sunshine.csv")
temp = pd.read_csv("Dataset/temp.csv")
wind = pd.read_csv("Dataset/wind.csv")

train_data = sunshine.merge(temp, on=['Day', 'Hour'], how='left')
train_data = train_data.merge(wind, on=['Day', 'Hour'], how='left')

sunshine1 = sunshine.iloc[0:150, :]
sunshine1['Day'] = sunshine1['Day'].map(lambda x: x + 300)

val_data = sunshine1.merge(temp, on=['Day', 'Hour'], how='left')
val_data = val_data.merge(wind, on=['Day', 'Hour'], how='left')

features = [f for f in train_data.columns if f not in ['Radiation', 'Day']]
X_train, X_test, Y_train, Y_test = train_test_split(train_data[features], train_data['Radiation'].values, test_size=0.1, random_state=42)

params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': args.num_leaves,
    'learning_rate': args.learning_rate,
    'metric': {'mse'},
    'verbose': -1,
    'min_data_in_leaf': args.min_data_in_leaf,
    'max_depth': args.max_depth
}

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

# train
gbm = lgb.train(params,
                train_set=lgb_train,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=30)])
                
Y_pred = gbm.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)

wandb.log({"mse": mse})