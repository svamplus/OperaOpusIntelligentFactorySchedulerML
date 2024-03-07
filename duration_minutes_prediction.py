import pickle
from typing import List
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import os
import optuna
import catboost
from optuna.importance import get_param_importances
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

from duration_in_minutes.data_utils import calculate_group_means

if os.path.exists("data/duration_in_minutes_history_enrich_train.csv"):
    data = pd.read_csv("data/duration_in_minutes_history_enrich_train.csv")
else:
    exit(1)
    # Connect to SQL Server
    server_name = 'ooaps.database.windows.net'
    database_name = 'ooAPS'
    username = 'XXXXX'
    password = 'XXXXXX'

    cnxn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}".format(
        server_name=server_name,
        database_name=database_name,
        username=username,
        password=password
    ))

    server_name = 'ooaps.database.windows.net'
    database_name = 'ooAPS'
    username = 'XXXXX'
    password = 'XXXXXX'

    cnxn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}".format(
        server_name=server_name,
        database_name=database_name,
        username=username,
        password=password
    ))

    # Retrieve data from table
    data = pd.read_sql_query("""
    SELECT [EmployeeEmploymentDuration] ,[EmployeeAge] ,[ResourceCode] ,[ResourceName] ,[FinalProductCode] ,[FinalProductName] ,[OperationCode] ,[OperationDescription] ,[DurationInMinutes] ,[Id] 
    FROM [ml].[OperationDurationHistory]""", cnxn)

# Preprocess data
target_col = 'DurationInMinutes'
cat_features = ['ResourceNameNorm', 'FinalProductNameNorm', 'OperationDescriptionNorm']
for cf in cat_features:
    data[cf] = data[cf].astype("str")
drop_columns = [target_col, 'ResourceName', 'FinalProductName', 'OperationDescription', 'Id', "Unnamed: 0", 'ResourceCode', 'FinalProductCode', 'OperationCode']
drop_columns.extend([c for c in data.columns if c.find("_EMB_") >= 0])
data_columns = data.columns

categorical_columns = list(data.select_dtypes(include=['object']).columns.values.tolist())
numeric_columns = list(data.select_dtypes(exclude=['object']).columns.values.tolist())
categorical_columns = list(set(categorical_columns).difference(drop_columns).union(cat_features))
numeric_columns = list(set(numeric_columns).difference(drop_columns).difference(cat_features))
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
    ('DurationMeansCalc', FunctionTransformer(calculate_group_means, validate=False),
     ['ResourceNameNorm', 'FinalProductNameNorm', 'OperationDescriptionNorm', target_col]),
    ("imputerN", SimpleImputer(strategy="median", missing_values=pd.NA), numeric_columns),
    ("imputerC", SimpleImputer(strategy="constant", missing_values=pd.NA, fill_value="Unknown"), categorical_columns),
    #('Embeddings', FunctionTransformer(get_sentence_embeddings_column, validate=False, kw_args={"dictionary":dictionary}),
    # ['Unnamed: 0']),
], remainder="drop", verbose_feature_names_out=False)
feature_engineering_cleaning.set_output(transform='pandas')
y = data[target_col]
X_trans = feature_engineering_cleaning.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.15, random_state=42)

cols:List[str] = X_train.columns
feature_weights = {c:(1.0 if c.find("EMB_") < 0 else 0.5) for c in cols} 

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
    depth = trial.suggest_int("depth", 8, 12),
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
        loss_function="MAPE",
        feature_weights=feature_weights
        #task_type="GPU",
        #devices='0:1'
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)

    err = np.amin(model.evals_result_['validation']['MAPE'])
    if err < best_error:
        best_error = err
        best_iter = model.tree_count_
    return err


study = optuna.create_study(sampler = optuna.samplers.CmaEsSampler(), direction="minimize")
study.optimize(objective, n_trials=500, show_progress_bar=True)
print(get_param_importances(study))


bp = study.best_params

#{'depth': 7, 'learning_rate': 0.8524753668685885, 'l2_leaf_reg': 0.7009668830601328, 'model_size_reg': 0.002988210500875645} 987
model = catboost.CatBoostRegressor(
        iterations=best_iter, 
        learning_rate=bp["learning_rate"], 
        bagging_temperature=bp["bagging_temperature"],
        depth=bp["depth"],
        random_strength=bp["random_strength"],
        l2_leaf_reg=bp["l2_leaf_reg"],
        model_size_reg=bp["model_size_reg"],
        cat_features=cat_features,
        loss_function="MAPE",
        feature_weights=feature_weights
)
model.fit(X_trans, y, early_stopping_rounds=30)
model.save_model("first_model_simple.cb")
joblib.dump(feature_engineering_cleaning, "first_model_simple.pipeline")
importances = model.feature_importances_

importances_sort_idx = np.argsort(-importances)
importances_sorted = importances[importances_sort_idx]
important_cols = X_trans.columns[importances_sort_idx]
importance_dict = {cname:impo for cname, impo in zip(important_cols, importances_sorted)}
print(importance_dict)
with open("first_model_simple.importances", "wb") as fw:
    pickle.dump(importance_dict, fw)
print(best_iter, best_error)
"""
X_train_trans = feature_engineering_cleaning.fit_transform(data)
cols:List[str] = X_train_trans.columns
feature_weights = {c:(1.0 if c.find("EMB_") < 0 else 0.5) for c in cols} 
assert "OrderedYear" not in cols

model_final = catboost.CatBoostRegressor(
        iterations=465, 
        learning_rate=0.06606963827346232, 
        #bagging_temperature=bp["bagging_temperature"],
        depth=10,
        #random_strength=bp["random_strength"],
        l2_leaf_reg=0.0025551140376089444,
        model_size_reg=0.006086804238457295,
        cat_features=cat_features,
        feature_weights=feature_weights,
        task_type="GPU",
        devices='0:1')


X_train_trans = X_train_trans[y > 0]
y_train = y[y > 0]
model_final.fit(X_train_trans, y_train)
model_final.save_model("latest_model.cb")
joblib.dump(feature_engineering_cleaning, "latest_model.pipeline")
importances = model_final.feature_importances_

importances_sort_idx = np.argsort(-importances)
importances_sorted = importances[importances_sort_idx]
important_cols = X_train_trans.columns[importances_sort_idx]

importance_dict = {cname:impo for cname, impo in zip(important_cols, importances_sorted)}
print(importance_dict)
with open("latest_model.importances", "wb") as fw:
    pickle.dump(importance_dict, fw)
"""