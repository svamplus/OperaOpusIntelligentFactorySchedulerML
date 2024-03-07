import pandas as pd
import itertools

def mean_encoding(
        df,
        categorical_columns = ["ItemGroup", "ItemType", "ResourceGroup", "StandardOperationCode"], 
        target_columns=["WorkerTimeDuration", "MachineTimeDuration", "Kol"], 
        cutoff=5
):
    #generate all possible combinatons of categorical columns
    combinations = []
    for i in range(1, len(categorical_columns)+1):
        combinations.extend(itertools.combinations(categorical_columns, i))
    means_dict = {}
    for k, combination in enumerate(combinations):
        comb_mean_dict_means = df.groupby(list(combination))[target_columns].mean().reset_index()
        comb_mean_dict_stds = df.groupby(list(combination))[target_columns].std().reset_index()
        #rename columns
        comb_mean_dict_means.rename(columns={t:t+'Mean' for t in target_columns}, inplace=True)
        comb_mean_dict_stds.rename(columns={t:t+'Std' for t in target_columns}, inplace=True)
        comb_mean_dict = pd.concat([comb_mean_dict_means, comb_mean_dict_stds[[t+'Std' for t in target_columns]]], axis=1)
        comb_mean_dict_counts = df.groupby(list(combination))[target_columns[0]].count().reset_index()
        comb_mean_dict = comb_mean_dict[comb_mean_dict_counts[target_columns[0]] > cutoff]
        means_dict[combination] = (comb_mean_dict, len(combination), k)
    means_dict["DefaultMean"] = df[target_columns].mean()
    means_dict["DefaultStd"] = df[target_columns].std()
    return means_dict