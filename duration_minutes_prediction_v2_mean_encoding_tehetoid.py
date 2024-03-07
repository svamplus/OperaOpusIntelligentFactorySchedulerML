from collections import OrderedDict
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import AsyncResult
import pickle
from typing import List
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import numpy as np

from duration_in_minutes.data_utils import get_month, get_weekday
from stratify_data_and_add_mean_encodings import enrich_with_mean_encodings 
from mean_encoding import mean_encoding
from stratify_tehetoid_ml_data import stratify_data

def fold_and_train(X, y, fold, kfolds, 
                   learning_rate, bagging_temperature, depth, random_strength, l2_leaf_reg, model_size_reg, 
                   loss_function, iterations, cat_features, text_columns, mode
):
    import catboost
    import numpy as np
    from stratify_data_and_add_mean_encodings import enrich_with_mean_encodings 
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from duration_in_minutes.data_utils import get_month
    from stratify_tehetoid_ml_data import stratify_data
    from mean_encoding import mean_encoding

    X = pd.concat([X, y], axis=1)
    target_cols = list(y.columns)
    X_train, X_test = stratify_data(X, test_size=0.15, random_state=fold*98345923874675)

    categorical_mean_dict_cols = ["TehETOID"]
    target_mean_dict_cols = ["WorkerTimeDuration", "Kol"]

    mean_encoding_dict = mean_encoding(X_train, categorical_mean_dict_cols, target_mean_dict_cols, 1)

    y_train = X_train[target_cols]
    X_train = X_train.drop(columns=target_cols)
    y_test = X_test[target_cols]
    X_test = X_test.drop(columns=target_cols)

    drop_columns = [*target_cols, 'DatLans', 'ItemType', 'ItemGroup', 'ItemCode', 'ItemName', 'ResourceGroup', 'StandardOperationCode', 'StandardOperationName', "TehETOID"]
    drop_columns.extend(['MachineTimeDuration', 'MachineTimeDurationMean'])
    categorical_columns = list(X.select_dtypes(include=['object']).columns.values.tolist())
    numeric_columns = list(X.select_dtypes(exclude=['object']).columns.values.tolist())
    categorical_columns = list(set(categorical_columns).difference(drop_columns).union(cat_features))
    numeric_columns = list(set(numeric_columns).difference(drop_columns).difference(cat_features))
    categorical_columns.remove("Month")
    categorical_columns.remove("Weekday")

    feature_engineering_cleaning = ColumnTransformer([
            ('EnrichData', FunctionTransformer(enrich_with_mean_encodings, validate=False, kw_args={"mean_encoding_dict":mean_encoding_dict, "leave_only_enrichments":True}), categorical_mean_dict_cols),
            ('MonthCalc', FunctionTransformer(get_month, validate=False),['DatLans']),
            ('WeekdayCalc', FunctionTransformer(get_weekday, validate=False),['DatLans']),
            ("imputerN", SimpleImputer(strategy="median", missing_values=pd.NA), numeric_columns),
            ("imputerC", SimpleImputer(strategy="constant", missing_values=pd.NA, fill_value=-1), categorical_columns),
            #('Embeddings', FunctionTransformer(get_sentence_embeddings_column, validate=False, kw_args={"dictionary":dictionary}),
            # ['Unnamed: 0']),
        ], remainder="drop", verbose_feature_names_out=False)
    feature_engineering_cleaning.set_output(transform='pandas')
    X_train_trans = feature_engineering_cleaning.fit_transform(X_train)
    X_test_trans = feature_engineering_cleaning.transform(X_test)

    if mode.find("residual") >= 0:
        y_train = pd.DataFrame(y_train.values - X_train_trans[[f"{c}Mean" for c in target_cols]].values, columns = target_cols, index=y_train.index)
        y_test = pd.DataFrame(y_test.values - X_test_trans[[f"{c}Mean" for c in target_cols]].values, columns = target_cols, index=y_test.index)

    y_train_mask = ~y_train.isna().any(axis=1)
    y_train = y_train[y_train_mask]
    X_train_trans = X_train_trans[y_train_mask]

    y_test_mask = ~y_test.isna().any(axis=1)
    y_test = y_test[y_test_mask]
    X_test_trans = X_test_trans[y_test_mask]
    
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
        verbose=0
    )

    model.fit(X_train_trans, y_train, eval_set=(X_test_trans, y_test), early_stopping_rounds=60)
    #iters = max(, iters)
    err = np.amin(model.evals_result_['validation'][loss_function])
    #print(f"Iter {i} err {err} best iter {model.best_iteration_}")
    return err, model.best_iteration_

def save_model_and_pipeline(best_model, feature_engineering_cleaning, name, columns):
    best_model.save_model(f"{name}.cb")
    joblib.dump(feature_engineering_cleaning, f"{name}.pipeline")
    importances = best_model.feature_importances_

    importances_sort_idx = np.argsort(-importances)
    importances_sorted = importances[importances_sort_idx]
    important_cols = columns[importances_sort_idx]
    importance_dict = {cname:impo for cname, impo in zip(important_cols, importances_sorted)}
    with open(f"{name}.importances", "wb") as fw:
        pickle.dump(importance_dict, fw)
    
    return importance_dict

if __name__ == "__main__":
    
    if os.path.exists("data/TehETOID_for_machine_learning.csv"):
        data = pd.read_csv("data/TehETOID_for_machine_learning_train.csv", 
            dtype={"WorkerTimeDuration":np.float32,"MachineTimeDuration":np.float32, "Kol":np.float32})
        data_eval = pd.read_csv("data/TehETOID_for_machine_learning_eval.csv", 
            dtype={"WorkerTimeDuration":np.float32,"MachineTimeDuration":np.float32, "Kol":np.float32})
    else:
        exit(1)
        

    target_cols = ['WorkerTimeDuration']
    modes = ["value_pred_std"]
    for mode in modes:
        train_data = data.copy()
        eval_data = data_eval.copy()
        model_name = f"models/new_minutes_durations_history_mape_{mode}"

        categorical_mean_dict_cols = ["TehETOID"]
        target_mean_dict_cols = ["WorkerTimeDuration", "Kol"]
        mean_encoding_dict = mean_encoding(train_data, categorical_mean_dict_cols, target_mean_dict_cols, 1)
        train_data_enrichment = enrich_with_mean_encodings(train_data[categorical_mean_dict_cols], mean_encoding_dict)
        
        cat_features = []
        drop_columns = [*target_cols, 'DatLans', 'ItemType', 'ItemGroup', 'ItemCode', 'ItemName', 'ResourceGroup', 'StandardOperationCode', 'StandardOperationName']
        drop_columns.extend(['MachineTimeDuration', 'MachineTimeDurationMean'])
        if not mode.endswith("std"):
            drop_columns.extend(["WorkerTimeDurationStd", "MachineTimeDurationStd", "KolStd"])
        text_columns = []

        categorical_columns = list(data.select_dtypes(include=['object']).columns.values.tolist())
        numeric_columns = list(data.select_dtypes(exclude=['object']).columns.values.tolist())
        categorical_columns = list(set(categorical_columns).difference(drop_columns).union(cat_features))
        numeric_columns = list(set(numeric_columns).difference(drop_columns).difference(cat_features))
        #categorical_columns.remove("Month")
        feature_engineering_cleaning = ColumnTransformer([
            ('EnrichData', FunctionTransformer(enrich_with_mean_encodings, validate=False, kw_args={"mean_encoding_dict":mean_encoding_dict, "leave_only_enrichments":True}), categorical_mean_dict_cols),
            ('MonthCalc', FunctionTransformer(get_month, validate=False),['DatLans']),
            ('WeekdayCalc', FunctionTransformer(get_weekday, validate=False),['DatLans']),
            ("imputerN", SimpleImputer(strategy="median", missing_values=pd.NA), [nc for nc in numeric_columns if nc != 'TehETOID']),
            ("imputerC", SimpleImputer(strategy="constant", missing_values=pd.NA, fill_value=-1), categorical_columns),
            #('Embeddings', FunctionTransformer(get_sentence_embeddings_column, validate=False, kw_args={"dictionary":dictionary}),
            # ['Unnamed: 0']),
        ], remainder="drop", verbose_feature_names_out=False)

        feature_engineering_cleaning_simple = ColumnTransformer([
            ("imputerN", SimpleImputer(strategy="median", missing_values=pd.NA), numeric_columns),
            ("imputerC", SimpleImputer(strategy="constant", missing_values=pd.NA, fill_value=-1), categorical_columns),
            ("pass", "passthrough", ["DatLans"])
            #('Embeddings', FunctionTransformer(get_sentence_embeddings_column, validate=False, kw_args={"dictionary":dictionary}),
            # ['Unnamed: 0']),
        ], remainder="drop", verbose_feature_names_out=False)
        feature_engineering_cleaning.set_output(transform='pandas')
        y_train = train_data[target_cols]
        X_train = feature_engineering_cleaning.fit_transform(train_data)
        X_eval = feature_engineering_cleaning.transform(eval_data)
        y_eval = eval_data[target_cols]

        y_train_mask = ~y_train[target_cols].isna().any(axis=1)
        X_train = X_train[y_train_mask]

        feature_engineering_cleaning_simple.set_output(transform='pandas')
        X_train_simple = feature_engineering_cleaning_simple.fit_transform(train_data)
        X_eval_simple = feature_engineering_cleaning.transform(eval_data)

        cols:List[str] = X_train.columns
        print(cols)

        cat_features.append("Month")
        cat_features.append("Weekday")
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
                            [X_train_simple, y_train, i, kfolds, 
                            learning_rate, bagging_temperature, depth, random_strength, l2_leaf_reg, model_size_reg, 
                            loss_function, iterations, cat_features,text_columns, mode]
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
                        text_features=text_columns,
                        silent=True
                        #task_type="GPU",
                        #devices='0:1'
                    )
                    
                    y_to_train = y_train[y_train_mask]
                    if mode.find("residual") >= 0:
                        y_to_train = pd.DataFrame(y_to_train.values - X_train[[f"{c}Mean" for c in target_cols]].values, columns = target_cols, index=y_to_train.index)
                    model.fit(X_train, y_to_train)
                    best_error = err
                    best_iter = model.tree_count_
                    best_model = model
                    with open(model_name + '_bp.txt', "w") as fw:
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
            y_eval_mask = ~y_eval.isna().any(axis=1)
            y_eval = y_eval[y_eval_mask]
            y_pred = y_pred[y_eval_mask]
            #print RMSE, MAE, MAPE between y_eval and y_pred, y_eval and y_pred have 2 columns
            mape = mean_absolute_percentage_error(y_eval.values, y_pred)
            mae = mean_absolute_error(y_eval.values, y_pred)
            rmse = np.sqrt(mean_squared_error(y_eval.values, y_pred))
            #print measure labels and values interchangably
            measure_labels = ['RMSE', 'MAE', 'MAPE']
            measure_values = [rmse, mae, mape]

            for label, value in zip(measure_labels, measure_values):
                print(f"Eval {label}: {value}")
    