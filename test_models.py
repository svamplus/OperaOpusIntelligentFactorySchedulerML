#%%
import catboost
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import seaborn as sns

def residual_plot(data_plot, y_target_name, y_pred_name, legend_train, legend_eval):
    sns.set_style("whitegrid")
    ax = sns.residplot(data=data_plot[data_plot["DATASET"] == 'Train'], x=y_target_name, y=y_pred_name, color=(0.2,0.2,0.8), label=legend_train, scatter_kws={"alpha":0.2})
    ax = sns.residplot(data=data_plot[data_plot["DATASET"] == 'Eval'], x=y_target_name, y=y_pred_name, color=(0.2,0.8,0.2), label=legend_eval, scatter_kws={"alpha":0.2})
    ax.legend()
    return ax
#%%
if __name__ == '__main__':
    #%%
    model_name = "new_minutes_durations_history_value_pred_std"
    model_path = f"models/{model_name}.cb"
    charts_dir = f"charts/{model_name}_3"
    os.makedirs(charts_dir, exist_ok=True)
    target_cols = ["WorkerTimeDuration"]

    data_train = pd.read_csv("data/TehETOID_for_machine_learning_train.csv")
    #data_train = data_train[data_train["FOCUSDATE"] <= '2019-02-01']
    #data_train["MINS_D730_FUTURE"] = data_train["MINS_D730_FUTURE"] + data_train["MINS_D365_FUTURE"]
    data_eval = pd.read_csv("data/TehETOID_for_machine_learning_eval.csv")
    #data_eval = data_eval[data_eval["FOCUSDATE"] <= '2019-02-01']
    #data_eval["MINS_D730_FUTURE"] = data_eval["MINS_D730_FUTURE"] + data_eval["MINS_D365_FUTURE"]

    numeric_columns = [c for c in data_train.columns if data_train.dtypes[c] == np.float64]
    data_train = data_train[~np.any(np.isinf(data_train[numeric_columns]), axis=1)]
    data_eval = data_eval[~np.any(np.isinf(data_eval[numeric_columns]), axis=1)]

    data_train["DATASET"] = 'Train'
    data_eval["DATASET"] = 'Eval'

    data = pd.concat([data_train, data_eval], axis=0).reset_index()

    pipeline_name = os.path.splitext(model_path)[0] + '.pipeline'
    model_final = catboost.CatBoostRegressor()
    model_final.load_model(model_path)

    transform = joblib.load(pipeline_name)
    y_target = data[target_cols]
    data_trans = transform.transform(data)
    y_pred = model_final.predict(data_trans).reshape(-1, len(target_cols))
    data_plot = data.assign(**{(target_col+"Pred"):y_pred[:,i] for i, target_col in enumerate(target_cols)})
    dcp = data_trans.copy()
    dcp["TehETOID"] = data["TehETOID"]
    dcp = dcp.assign(**{(target_col+"Pred"):y_pred[:,i] for i, target_col in enumerate(target_cols)})
    dcp[target_cols] = y_target
    dcp["DATASET"] = data["DATASET"]
    dcp.to_csv(os.path.join(charts_dir, "predictions.csv"), index=False)
    if "residual" in model_name:
        for target_col in target_cols:
            data_plot[target_col+"Pred"] = data_plot[target_col+"Pred"] + data_trans[target_col+"Mean"]
    #additional extra old * 1.10474 - 123.544
    for i, target_col in enumerate(target_cols):
        with open(os.path.join(charts_dir, "metrics.txt"), "w") as fw:
            print("NAME", target_col, model_name, file=fw)
            y_eval_data = y_target.loc[data["DATASET"]=='Eval', target_col]
            y_eval_pred = y_pred[data["DATASET"]=='Eval'][:,i]
            y_train_data = y_target.loc[data["DATASET"]=='Train', target_col]
            y_train_pred = y_pred[data["DATASET"]=='Train'][:,i]

            if "residual" in model_name:
                y_eval_pred = y_eval_pred + data_trans.loc[data["DATASET"]=='Eval', target_col+"Mean"]
                y_train_pred = y_train_pred + data_trans.loc[data["DATASET"]=='Train', target_col+"Mean"]
            
            y_eval_data = y_target.loc[data["DATASET"]=='Eval', target_col]
            y_eval_data_mask = ~y_eval_data.isna()
            y_eval_pred = y_pred[data["DATASET"]=='Eval'][:,i]
            y_eval_data = y_eval_data[y_eval_data_mask]
            y_eval_pred = y_eval_pred[y_eval_data_mask]

            print("RMSE",np.sqrt(mean_squared_error(y_eval_data, y_eval_pred)), file=fw)
            print("MAPE",mean_absolute_percentage_error(y_eval_data, y_eval_pred), file=fw)
            r2_eval = r2_score(y_eval_data, y_eval_pred)
            legend_eval = f"Eval R2 {r2_eval:.3f}"
            print("R2",r2_eval, file=fw)
            print("==================================", file=fw)

            y_train_data = y_target.loc[data["DATASET"]=='Train', target_col]
            y_train_data_mask = ~y_train_data.isna()
            y_train_pred = y_pred[data["DATASET"]=='Train'][:,i]
            y_train_data = y_train_data[y_train_data_mask]
            y_train_pred = y_train_pred[y_train_data_mask]

            print("RMSE Train",np.sqrt(mean_squared_error(y_train_data, y_train_pred)), file=fw)
            print("MAPE Train",mean_absolute_percentage_error(y_train_data, y_train_pred), file=fw)
            r2_train = r2_score(y_train_data,y_train_pred)
            legend_train = f"Train R2 {r2_train:.3f}"
            print("R2 Train",r2_train, file=fw)
            print("==================================", file=fw)
        
        #%%
        ax1 = residual_plot(data_plot, target_col, target_col+"Pred", legend_train, legend_eval)
        plt.savefig(os.path.join(charts_dir, f"residual_plot_{target_col}.png"))
        plt.clf()
    #%%
    import shap

    explainer = shap.TreeExplainer(model_final)
    shap_values = explainer.shap_values(data_trans)
    #shap_values = model_final.get_feature_importance(
    #    catboost.Pool(data_trans, y_target,
    #                  cat_features=[model_final.feature_names_[i] for i in model_final.get_cat_feature_indices()]), type='ShapValues')

    fig = shap.summary_plot(shap_values, data_trans, show=False)
    plt.savefig(os.path.join(charts_dir, "feature_impacts.png"))
    plt.clf()

    is_shap_list = isinstance(shap_values, list)
    if is_shap_list:
        for i, target_col in enumerate(target_cols):
            fig = shap.summary_plot(shap_values[i], data_trans, show=False)
            plt.savefig(os.path.join(charts_dir, f"{target_col}_feature_impacts.png"))
            plt.clf()

    worker_time_duration_shap = shap_values[target_cols.index("WorkerTimeDuration")] if is_shap_list else shap_values
    if "Month" in data_trans.columns:
        shap.dependence_plot("Month", worker_time_duration_shap, data_trans, show=False, interaction_index="WorkerTimeDurationMean")
        plt.savefig(os.path.join(charts_dir, "Month_dependence_plot.png"))
        plt.clf()
    if "Weekday" in data_trans.columns:
        shap.dependence_plot("Weekday", worker_time_duration_shap, data_trans, show=False, interaction_index="WorkerTimeDurationMean")
        plt.savefig(os.path.join(charts_dir, "Weekday_dependence_plot.png"))
        plt.clf()
    # arr = np.abs(shap_values).mean(0)
    # corrs = data_trans.corrwith(pd.DataFrame(shap_values, columns = data_trans.columns)).values.round(2)
    # indices = np.argsort(-arr)

    # columns = data_trans.columns

    # sorted_columns = columns[indices]
    # sorted_values = arr[indices]

    # arr_vals = []
    # col_names = []
    # arr_vals.append(arr[indices])
    # arr_vals.append(corrs[indices])
    # col_names.append('SHAP_IMPACT')
    # col_names.append('SHAP_DIRECTION')
    # stacked = np.stack([sorted_columns, *arr_vals, sorted_values], axis=1)
    # df = pd.DataFrame(stacked, columns=["FEATURE", *col_names, "TOTAL_SHAP_IMPACT"])
    # df.to_csv(os.path.join(charts_dir, "shap_summary_abs.csv"), index=False)

