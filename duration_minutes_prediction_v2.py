from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import AsyncResult
import pickle
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import pyodbc

import os
import optuna
import catboost
from optuna.importance import get_param_importances
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import numpy as np
from functools import partial

from duration_in_minutes.data_utils import calculate_group_means_2, get_month

if os.path.exists("data/duration_in_minutes_history_2.csv"):
    data = pd.read_csv("data/duration_in_minutes_history_2.csv")
else:
    exit(1)
    # Connect to SQL Server
    server_name = 'ooaps.database.windows.net'
    database_name = 'ooAPS'
    username = 'XXXXX'
    password = 'XXXXX'

    cnxn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}".format(
        server_name=server_name,
        database_name=database_name,
        username=username,
        password=password
    ))

    server_name = 'ooaps.database.windows.net'
    database_name = 'ooAPS'
    username = 'XXXXX'
    password = 'XXXXX'

    cnxn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}".format(
        server_name=server_name,
        database_name=database_name,
        username=username,
        password=password
    ))

    # Retrieve data from table
    data = pd.read_sql_query("""
    SELECT * 
    FROM [ml].[OperationDurationHistory2]
    """, cnxn)

# Preprocess data
target_cols = ['WorkerTimeDuration','ResourceTimeDuration']
cat_features = ['ItemType','ItemGroup','ItemCode','ResourceGroup','StandardOperationCode',]
text_columns=['ItemName', 'StandardOperationName']
for cf in cat_features:
    data[cf] = data[cf].astype("str")
drop_columns = [*target_cols, 'DateStarted']
drop_columns.extend([c for c in data.columns if c.find("_EMB_") >= 0])
data_columns = data.columns

categorical_columns = list(data.select_dtypes(include=['object']).columns.values.tolist())
numeric_columns = list(data.select_dtypes(exclude=['object']).columns.values.tolist())
categorical_columns = list(set(categorical_columns).difference(drop_columns).union(cat_features))
numeric_columns = list(set(numeric_columns).difference(drop_columns).difference(cat_features))
cat_features.append("Month")
#numeric_columns += [target_col+'_NORM']

last_step_columns = set(data_columns).difference(drop_columns)

feature_engineering_cleaning = ColumnTransformer([
    #('ForeignCountry', FunctionTransformer(get_foreign_country_column, validate=False),
     #['SupplierCountry']),
    #('ItemGroupSub', FunctionTransformer(get_item_group_sub_column, validate=False),
    # ['ItemGroup']),
    #('Weekday', FunctionTransformer(get_weekday_column, validate=False),
    # ['OrderedYear', 'OrderedMonth', 'OrderedDay']),
    #("drop", "drop", ["OrderedYear"]),
    ('WorkerDurationMeansCalc', FunctionTransformer(partial(calculate_group_means_2, target_column_name="WorkerTimeDuration"), validate=False),
     ['ItemType','ItemGroup','ItemCode','ResourceGroup','StandardOperationCode', target_cols[0]]),
    ('ResourceDurationMeansCalc', FunctionTransformer(partial(calculate_group_means_2, target_column_name="ResourceTimeDuration"), validate=False),
     ['ItemType','ItemGroup','ItemCode','ResourceGroup','StandardOperationCode', target_cols[1]]),
     ('MonthCalc', FunctionTransformer(get_month, validate=False),
     ['DateStarted']),
    ("imputerN", SimpleImputer(strategy="median", missing_values=pd.NA), numeric_columns),
    ("imputerC", SimpleImputer(strategy="constant", missing_values=pd.NA, fill_value="Unknown"), categorical_columns),
    #('Embeddings', FunctionTransformer(get_sentence_embeddings_column, validate=False, kw_args={"dictionary":dictionary}),
    # ['Unnamed: 0']),
], remainder="drop", verbose_feature_names_out=False)
feature_engineering_cleaning.set_output(transform='pandas')
y = data[target_cols]
X_trans = feature_engineering_cleaning.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.15, random_state=42)

cols:List[str] = X_train.columns
feature_weights = {c:(1.0 if c.find("EMB_") < 0 else 0.5) for c in cols} 
print(cols)

#X_train = feature_engineering_cleaning.fit_transform(X_train)
"""
model = catboost.CatBoostRegressor(
        iterations=1000, 
        learning_rate=3e-2, 
        bagging_temperature=0.4,
        depth=6,
        random_strength=1.2,
        cat_features=cat_features)

model.fit(X_train, y_train, eval_set=(X_test, y_test))
"""

def detuplize(value):
    return value[0] if isinstance(value, tuple) else value

best_iter = 1000
best_error = 99999
def objective(trial: optuna.Trial):
    global best_error
    global best_iter
    depth = trial.suggest_int("depth", 3, 8),
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True),
    random_strength = trial.suggest_float("random_strength", 1e-3, 10, log=True),
    bagging_temperature = trial.suggest_float("bagging_temperature", 1e-3, 10, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-4, 1, log=True)
    model_size_reg = trial.suggest_float("model_size_reg", 1e-4, 1, log=True)

    model = catboost.CatBoostRegressor(
        iterations=1000, 
        learning_rate=detuplize(learning_rate), 
        bagging_temperature=detuplize(bagging_temperature),
        depth=detuplize(depth),
        random_strength=detuplize(random_strength),
        l2_leaf_reg=detuplize(l2_leaf_reg),
        model_size_reg=detuplize(model_size_reg),
        cat_features=cat_features,
        loss_function="MultiRMSE",
        feature_weights=feature_weights,
        text_features=text_columns
        #task_type="GPU",
        #devices='0:1'
    )

                model.fit(X_train, y_train)
                best_error = err
                best_iter = model.tree_count_
                best_model = model
                with open(os.path.join("models", model_name + '_bp.txt'), "w") as fw:
                    print(OrderedDict({}), {
                        "depth": depth,
                        "learning_rate":learning_rate,
                        "random_strength":random_strength,
                        "bagging_temperature":bagging_temperature,
                        "l2_leaf_reg":l2_leaf_reg,
                        "model_size_reg":model_size_reg
                    }, best_iter, file=fw, flush=True)
                save_model_and_pipeline(model, feature_engineering_cleaning,model_name,X_train.columns)
            return err



        study = optuna.create_study(sampler = optuna.samplers.CmaEsSampler(), direction="minimize")
        study.optimize(objective, n_trials=500, show_progress_bar=True)
        print(get_param_importances(study))

        bp = study.best_params
        with open(model_name + '_bp.txt', "w") as fw:
            print(get_param_importances(study), bp, best_iter, file=fw, flush=True)

        #{'depth': 7, 'learning_rate': 0.8524753668685885, 'l2_leaf_reg': 0.7009668830601328, 'model_size_reg': 0.002988210500875645} 987
        model = best_model
        importance_dict = save_model_and_pipeline(model, feature_engineering_cleaning,model_name, X_eval.columns)

        print(importance_dict)
        with open(f"{model_name}.importances", "wb") as fw:
            pickle.dump(importance_dict, fw)
        print(bp, best_iter, best_error)

        y_pred = best_model.predict(X_eval)
        #print RMSE, MAE, MAPE between y_eval and y_pred, y_eval and y_pred have 2 columns
        mape = mean_absolute_percentage_error(y_eval.values, y_pred)
        mae = mean_absolute_error(y_eval.values, y_pred)
        rmse = np.sqrt(mean_squared_error(y_eval.values, y_pred))
        #print measure labels and values interchangably
        measure_labels = ['RMSE', 'MAE', 'MAPE']
        measure_values = [rmse, mae, mape]

        for label, value in zip(measure_labels, measure_values):
            print(f"Eval {label}: {value}")
    