# Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

sunshine = pd.read_csv("Dataset/sunshine.csv")
temp = pd.read_csv("Dataset/temp.csv")
wind = pd.read_csv("Dataset/wind.csv")

train_data = sunshine[0:4350].merge(temp, on=['Day', 'Hour'], how='left')
train_data = train_data.merge(wind, on=['Day', 'Hour'], how='left')


train_data['day_t'] = train_data['Day']
train_data_temp = train_data.copy()
train_data_temp['day_t'] = train_data_temp['Day'].map(lambda x: x + 7)
train_data_temp.rename(columns={'Radiation': 'Radiation_before'}, inplace=True)
train_data_temp = train_data_temp[['day_t', 'Radiation_before', 'Hour']]
train_data = train_data.merge(train_data_temp, on=['day_t', 'Hour'], how='left')

sunshine1 = sunshine.iloc[0:150, :]
sunshine1['Day'] = sunshine1['Day'].map(lambda x: x + 300)

val_data = sunshine1.merge(temp, on=['Day', 'Hour'], how='left')
val_data = val_data.merge(wind, on=['Day', 'Hour'], how='left')
val_data['day_t'] = val_data['Day']
val_data_temp = val_data.copy()
val_data_temp['day_t'] = val_data_temp['Day'].map(lambda x: x + 7)
val_data_temp.rename(columns={'Radiation': 'Radiation_before'}, inplace=True)
val_data_temp = val_data_temp[['day_t', 'Radiation_before', 'Hour']]
val_data = val_data.merge(val_data_temp, on=['day_t', 'Hour'], how='left')

features = [f for f in train_data.columns if f not in ['Radiation', 'day_t']]
X_train, X_test, Y_train, Y_test = train_test_split(train_data[features], train_data['Radiation'].values, test_size=0.1, random_state=42)

params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 11,
    'learning_rate': 0.1,
    'metric': {'mse'},
    'verbose': -1,
    'min_data_in_leaf': 4,
    'max_depth':5
}

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

# train
gbm = lgb.train(params,
                train_set=lgb_train,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=30)])

# 预测提交数据                
Y_pred = gbm.predict(val_data[features])

dataframe = pd.DataFrame({'Radiation': Y_pred})
dataframe.to_csv("Dataset/Y_pred.csv", index=False, sep=',')