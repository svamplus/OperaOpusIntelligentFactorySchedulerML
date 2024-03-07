import pandas as pd
import os
import numpy as np
#from data_utils import normalize_product_column, normalize_operation_column

file_path = "data/duration_in_minutes_history_enrich.csv"
data = pd.read_csv(file_path)
data = data[(data["DurationInMinutes"] < 800) & (data["DurationInMinutes"] > 0)]
groups = data.groupby(["ResourceNameNorm", "FinalProductNameNorm", "OperationDescriptionNorm"])
relevant_groups_data = groups.filter(lambda g: g["Id"].count() >= 60)
fc_sample_eval = relevant_groups_data.groupby(["ResourceNameNorm", "FinalProductNameNorm", "OperationDescriptionNorm"]).sample(n=20)
fc_sample_train = data.loc[relevant_groups_data.index]
fc_sample_train = fc_sample_train.loc[fc_sample_train.index.difference(fc_sample_eval.index)]
fc_sample_train.to_csv("data/duration_in_minutes_history_enrich_train.csv", index=False)
fc_sample_eval.to_csv("data/duration_in_minutes_history_enrich_eval.csv", index=False)