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

from duration_in_minutes.data_utils import integrate_group_means, get_month
from mean_encoding import mean_encoding

def fold_and_train(X, y, fold, kfolds, 
                   learning_rate, bagging_temperature, depth, random_strength, l2_leaf_reg, model_size_reg, 
                   loss_function, iterations, cat_features, text_columns
):
    import catboost
    import numpy as np
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=fold*98345923874675)
    
    model = catboost.CatBoostRegressor(
        iterations=iterations, 
        learning_rate=learning_rate, 
        bagging_temperature=bagging_temperature,
        depth=depth,
        random_strength=random_strength,
        l2_leaf_reg=l2_leaf_reg,
        model_size_reg=model_size_reg,
        cat_features=cat_features,
        text_features=text_columns,
        loss_function=loss_function,
        verbose=0,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=60)
    #iters = max(, iters)
    err = np.amin(model.evals_result_['validation'][loss_function])
    #print(f"Iter {i} err {err} best iter {model.best_iteration_}")
    return err, model.best_iteration_

def save_model_and_pipeline(best_model, feature_engineering_cleaning, name, columns):
    best_model.save_model(f"models/{name}.cb")
    joblib.dump(feature_engineering_cleaning, f"models/{name}.pipeline")
    importances = best_model.feature_importances_

    importances_sort_idx = np.argsort(-importances)
    importances_sorted = importances[importances_sort_idx]
    important_cols = columns[importances_sort_idx]
    importance_dict = {cname:impo for cname, impo in zip(important_cols, importances_sorted)}
    with open(f"{name}.importances", "wb") as fw:
        pickle.dump(importance_dict, fw)
    
    return importance_dict

if __name__ == "__main__":
    
    if os.path.exists("data/duration_in_minutes_history_2.csv"):
        data = pd.read_csv("data/new_minutes_durations_history_mean_enriched_train.csv", 
            dtype={"WorkerTimeDuration":np.float32,"MachineTimeDuration":np.float32, "Kol":np.float32, "KolMean":np.float32, "KolStd":np.float32,
                   "WorkerTimeDurationMean":np.float32, "WorkerTimeDurationStd":np.float32, "MachineTimeDurationMean":np.float32, "MachineTimeDurationStd":np.float32})
        data_eval = pd.read_csv("data/new_minutes_durations_history_mean_enriched_eval.csv", 
            dtype={"WorkerTimeDuration":np.float32,"MachineTimeDuration":np.float32, "Kol":np.float32, "KolMean":np.float32, "KolStd":np.float32,
                   "WorkerTimeDurationMean":np.float32, "WorkerTimeDurationStd":np.float32, "MachineTimeDurationMean":np.float32, "MachineTimeDurationStd":np.float32})
    else:
        exit(1)
        # Connect to SQL Server
    # server_name = 'ooaps.database.windows.net'
    # database_name = 'ooAPS'
    # username = 'svam_sql'
    # password = '#oo!SP1024'

    # cnxn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}".format(
    #     server_name=server_name,
    #     database_name=database_name,
    #     username=username,
    #     password=password
    # ))

    # server_name = 'ooaps.database.windows.net'
    # database_name = 'ooAPS'
    # username = 'svam_sql'
    # password = '#oo!SP1024'

    # cnxn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}".format(
    #     server_name=server_name,
    #     database_name=database_name,
    #     username=username,
    #     password=password
    # ))

    # # Retrieve data from table
    # data = pd.read_sql_query("""
    # SELECT [TehETOID]
    #   ,[DatLans]
    #   ,[Kol]
    #   ,[ItemType]
    #   ,[ItemGroup]
    #   ,[ItemCode]
    #   ,[ItemName]
    #   ,[ResourceGroup]
    #   ,[StandardOperationCode]
    #   ,[StandardOperationName]
    #   ,[WorkerTimeDuration]
    #   ,[MachineTimeDuration]
    # FROM [dbo].[OperationDurationHistory_20240202]
    # """, cnxn)

    # Preprocess data
        

    target_cols = ['WorkerTimeDuration','MachineTimeDuration']
    modes = ["value_pred", "residual_pred", "value_pred_std", "residual_pred_std"]
    for mode in modes:
        train_data = data.copy()
        eval_data = data_eval.copy()
        model_name = f"models/new_minutes_durations_history_mape_{mode}"

        if mode == "residual_pred":
            data["WorkerTimeDuration"] = data["WorkerTimeDuration"] - data["WorkerTimeDurationMean"]
            data["MachineTimeDuration"] = data["MachineTimeDuration"] - data["MachineTimeDurationMean"]
        
        cat_features = ["LOD"]
        drop_columns = [*target_cols, 'TehETOID', 'DatLans', 'ItemType', 'ItemGroup', 'ItemCode', 'ItemName', 'ResourceGroup', 'StandardOperationCode', 'StandardOperationName']
        if not mode.endswith("std"):
            drop_columns.extend(["WorkerTimeDurationStd", "MachineTimeDurationStd", "KolStd"])
        text_columns = []

        categorical_columns = list(data.select_dtypes(include=['object']).columns.values.tolist())
        numeric_columns = list(data.select_dtypes(exclude=['object']).columns.values.tolist())
        categorical_columns = list(set(categorical_columns).difference(drop_columns).union(cat_features))
        numeric_columns = list(set(numeric_columns).difference(drop_columns).difference(cat_features))

        feature_engineering_cleaning = ColumnTransformer([
            ('MonthCalc', FunctionTransformer(get_month, validate=False),
            ['DatLans']),
            ("imputerN", SimpleImputer(strategy="median", missing_values=pd.NA), numeric_columns),
            ("imputerC", SimpleImputer(strategy="constant", missing_values=pd.NA, fill_value=-1), categorical_columns),
            #('Embeddings', FunctionTransformer(get_sentence_embeddings_column, validate=False, kw_args={"dictionary":dictionary}),
            # ['Unnamed: 0']),
        ], remainder="drop", verbose_feature_names_out=False)
        feature_engineering_cleaning.set_output(transform='pandas')
        y_train = train_data[target_cols]
        X_train = feature_engineering_cleaning.fit_transform(train_data)
        X_eval = feature_engineering_cleaning.transform(eval_data)
        y_eval = eval_data[target_cols]

        cols:List[str] = X_train.columns
        print(cols)

        cat_features.append("Month")
        iterations = 2000
        best_iter = iterations
        best_error = 99999

        best_model:catboost.CatBoostRegressor = None

        with Pool(cpu_count() // 4) as pool:
            def objective(trial: optuna.Trial):
                global best_error
                global best_iter
                global best_model

                loss_function="MAPE"
                
                depth = trial.suggest_int("depth", 2, 8)
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
                random_strength = trial.suggest_float("random_strength", 1e-3, 10, log=True)
                bagging_temperature = trial.suggest_float("bagging_temperature", 1e-3, 10, log=True)
                l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-4, 1, log=True)
                model_size_reg = trial.suggest_float("model_size_reg", 1e-4, 1, log=True)

                kfolds = 8
                errs_async:List[AsyncResult] = []
                
                for i in range(kfolds):
                    errs_async.append(
                        pool.apply_async(
                            fold_and_train, 
                            [X_train, y_train, i, kfolds, 
                            learning_rate, bagging_temperature, depth, random_strength, l2_leaf_reg, model_size_reg, 
                            loss_function, iterations, cat_features,text_columns]
                            )
                        )
                    #errs_async.append(fold_and_train(data, i, kfolds, tcol, feature_engineering_cleaning, learning_rate, bagging_temperature, depth, random_strength, l2_leaf_reg, model_size_reg, loss_function, class_weights, iterations, cat_features))
                errs_iters = [e.get() for e in errs_async]
                errs, iters = zip(*errs_iters)
                err = sum(errs) / len(errs)

                iters = max(iters)
                if err < best_error:
                    model = catboost.CatBoostRegressor(
                        iterations=iters, 
                        learning_rate=learning_rate, 
                        bagging_temperature=bagging_temperature,
                        depth=depth,
                        random_strength=random_strength,
                        l2_leaf_reg=l2_leaf_reg,
                        model_size_reg=model_size_reg,
                        cat_features=cat_features,
                        loss_function=loss_function,
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
    