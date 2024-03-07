#%%
import pandas as pd
import os
import numpy as np
#from data_utils import normalize_product_column, normalize_operation_column

file_path = "data/duration_in_minutes_history.csv"
data = pd.read_csv(file_path)
data = data[(data["DurationInMinutes"] < 800) & (data["DurationInMinutes"] > 0)]
#%%
groups = data.groupby(["ResourceCode", "FinalProductCode", "OperationCode"])
relevant_groups_data = groups.filter(lambda g: g["Id"].count() >= 50)
fc_sample_eval = relevant_groups_data.groupby(["ResourceCode", "FinalProductCode", "OperationCode"]).sample(n=12)
#%%
fc_sample_train = data.loc[relevant_groups_data.index]
fc_sample_train = fc_sample_train.loc[fc_sample_train.index.difference(fc_sample_eval.index)]
#%%
fc_sample_train.to_csv("data/duration_in_minutes_history_train.csv", index=False)
fc_sample_eval.to_csv("data/duration_in_minutes_history_eval.csv", index=False)
"""
categories = data[["ResourceName", "OperationDescription", "FinalProductName"]].drop_duplicates()
print(len(categories))
categories = categories.sort_values(by=["ResourceName"])

normalised = normalize_product_column(data[["FinalProductName"]], "FinalProductName")
product_names_norm = pd.concat([data[["FinalProductName"]], normalised], axis=1).drop_duplicates()
product_names_norm.to_csv("product_names.csv", index=False)
print(len(normalised.drop_duplicates()), "/", len(product_names_norm))

data[["OperationDescription"]].drop_duplicates().to_csv("operation_descriptions.csv", index=False)
operation_normalised = normalize_operation_column(data[["OperationDescription"]], "OperationDescription")
operations_norm = pd.concat([data[["OperationDescription"]], operation_normalised], axis=1).drop_duplicates()
operations_norm.to_csv("operation_names.csv", index=False)
print(len(operation_normalised.drop_duplicates()), "/", len(operations_norm))

data[["ResourceName"]].drop_duplicates().to_csv("resource_descriptions.csv", index=False)
resource_normalised = normalize_product_column(data[["ResourceName"]], "ResourceName")
resource_norm = pd.concat([data[["ResourceName"]], resource_normalised], axis=1).drop_duplicates()
resource_norm.to_csv("resource_names.csv", index=False)
print(len(resource_normalised.drop_duplicates()), "/", len(resource_norm))

data = pd.concat([data, normalised, operation_normalised, resource_normalised], axis=1)
print(len(data[["ResourceNameNorm", "OperationDescriptionNorm", "FinalProductNameNorm"]].drop_duplicates()), "/", len(categories))
"""
#%%
"""
data["LeadTimeDaysStrata"] = data.apply(lambda row: lead_time_to_group(row["LeadTimeDays"]), axis = 1)
data["ForeignCountry"] = get_foreign_country_column(data)

mask = data["ItemGroupNorm"].isin(["Rotacijski silosni posipač d nastavak", "Ostali pribor", "Vijci s navojem do pola"])
eval_data_first = data[mask]
train_data_first = data[~mask]

foreign_country_data = train_data_first[train_data_first["ForeignCountry"] == 'Yes']
home_country_data = train_data_first[train_data_first["ForeignCountry"] == 'No']

fc_groups = foreign_country_data.groupby(["ItemGroupNorm", "LeadTimeDaysStrata"])
relevant_groups_data = fc_groups.filter(lambda g: g["ID"].count() >= 10)
fc_sample_eval = relevant_groups_data.groupby(["ItemGroupNorm", "LeadTimeDaysStrata"]).sample(n=6)
fc_sample_train = foreign_country_data.loc[foreign_country_data.index.difference(fc_sample_eval.index)]

hc_groups = home_country_data.groupby(["ItemGroupNorm", "LeadTimeDaysStrata"])
relevant_groups_data = hc_groups.filter(lambda g: g["ID"].count() >= 10)
hc_sample_eval = relevant_groups_data.groupby(["ItemGroupNorm", "LeadTimeDaysStrata"]).sample(n=6)
hc_sample_train = home_country_data.loc[home_country_data.index.difference(hc_sample_eval.index)]

data_train = pd.concat([hc_sample_train, fc_sample_train], axis=0)
data_eval = pd.concat([eval_data_first, fc_sample_eval, hc_sample_eval], axis=0)

data_train.to_csv("data/supply_lead_time_history_embed_train.csv", index=False)
data_eval.to_csv("data/supply_lead_time_history_embed_eval.csv", index=False)
"""