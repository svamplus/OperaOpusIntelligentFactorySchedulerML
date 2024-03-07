import pandas as pd
import pyodbc

import os
from optuna.importance import get_param_importances
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import catboost

from duration_in_minutes.data_utils import calculate_group_means_2, get_month

if os.path.exists("data/new_minutes_durations_history.csv"):
    data = pd.read_csv("data/new_minutes_durations_history.csv")
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
model_name = "new_minutes_durations_history_mape_value_pred_std"
model_path = f"models/{model_name}.cb"
target_cols = ["WorkerTimeDuration"]

pipeline_name = os.path.splitext(model_path)[0] + '.pipeline'
model_final = catboost.CatBoostRegressor()
model_final.load_model(model_path)
transform = joblib.load(pipeline_name)

init_means = pd.read_csv("data/TehETOID_means.csv", usecols=["TehETOID", "WorkerTimeDurationMean"])
init_means.rename(columns={"WorkerTimeDurationMean":"WorkerTimeDuration"}, inplace=True)
data.drop(columns=["WorkerTimeDuration"], inplace=True, errors="ignore")
data = data.merge(init_means, on="TehETOID", how="left")
pred_data_1 = data[~data["WorkerTimeDuration"].isna()]
data_to_pred = data[data["WorkerTimeDuration"].isna()].copy()
data_trans = transform.transform(data_to_pred)
y_pred = model_final.predict(data_trans)
data_to_pred["WorkerTimeDuration"] = y_pred
data_pred = pd.concat([pred_data_1, data_to_pred], axis=0).reset_index(drop=True)

data_pred.to_csv("data/new_minutes_durations_history_pred.csv", index=False)