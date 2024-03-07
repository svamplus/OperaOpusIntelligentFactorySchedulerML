from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Tuple
from mean_encoding import mean_encoding
import random
import joblib


def stratify_data(data:pd.DataFrame, test_size=0.2, random_state=2142567) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify the data based on the target variable."""
    return train_test_split(data, test_size=test_size, random_state=random_state)

def enrich_with_mean_encodings(data: pd.DataFrame, mean_encoding_dict:Dict[str, Tuple[pd.DataFrame, int, int]], leave_only_enrichments=True) -> pd.DataFrame:
    initial_len = data.shape[0]
    new_df = data.copy()
    new_df["LOD"] = -1
    order_dfs = sorted([(k, v[2], v[0], v[1]) for k, v in mean_encoding_dict.items() if not isinstance(k, str)], key=lambda x: x[1], reverse=True)
    #print("Total number of rows to merge:", len(new_df))
    start = True
    for columns_to_merge_on, _order, df_to_merge, lod in order_dfs:
        target_columns = [c for c in df_to_merge.columns if c not in columns_to_merge_on]
        merged = new_df.merge(df_to_merge, on=columns_to_merge_on, how="left")
        if not start:
            for c in target_columns:
                merged[c] = merged[c+"_x"].combine_first(merged[c+"_y"])
            merged.drop(columns=[c+"_x" for c in target_columns]+[c+"_y" for c in target_columns], inplace=True)
        not_merged_mask = merged[target_columns[0]].isna()
        filled_rows = (~not_merged_mask) & (merged["LOD"]==-1)
        merged.loc[filled_rows, "LOD"] = lod
        not_merged = not_merged_mask.sum()
        new_df = merged
        if not_merged == 0:
            break
        else:
            #print(f"Could not merge {columns_to_merge_on} with {target_columns} on {not_merged} rows.")
            start = False
    if not_merged > 0:
        default_means:pd.Series = mean_encoding_dict["DefaultMean"]
        #add means to names in index
        default_means.index = [c+"Mean" for c in default_means.index]
        default_stds = mean_encoding_dict["DefaultStd"]
        default_stds.index = [c+"Std" for c in default_means.index]
        mean_cols = [c for c in default_means.index if c in new_df.columns]
        new_df[mean_cols].fillna(default_means, inplace=True)
        std_cols = [c for c in default_stds.index if c in new_df.columns]
        new_df[std_cols].fillna(default_stds, inplace=True)
        new_df.loc[new_df["LOD"]==-1, "LOD"] = 0
    final_len = new_df.shape[0]
    assert initial_len == final_len, f"Initial length: {initial_len}, final length: {final_len}"
    new_df.index = data.index
    if leave_only_enrichments:
        return new_df.iloc[:, -4:]
    return new_df

def enrich_with_mean_encodings_random(data: pd.DataFrame, mean_encoding_dict:Dict[str, Tuple[pd.DataFrame, int, int]]) -> pd.DataFrame:
    new_df = data.copy()
    order_dfs = sorted([(k, v[2], v[0], v[1]) for k, v in mean_encoding_dict.items() if not isinstance(k, str)], key=lambda x: x[1], reverse=True)
    df_parts = len(order_dfs)
    #split df into equal df_parts
    dfs = [new_df.iloc[i::df_parts] for i in range(df_parts)]
    for i in range(df_parts):
        random.shuffle(order_dfs)
        df = dfs[i]
        df["LOD"] = -1
        
        #print("Total number of rows to merge:", len(new_df))
        start = True
        for columns_to_merge_on, _order, df_to_merge, lod in order_dfs:
            target_columns = [c for c in df_to_merge.columns if c not in columns_to_merge_on]
            merged = df.merge(df_to_merge, on=columns_to_merge_on, how="left")
            if not start:
                for c in target_columns:
                    merged[c] = merged[c+"_x"].combine_first(merged[c+"_y"])
                merged.drop(columns=[c+"_x" for c in target_columns]+[c+"_y" for c in target_columns], inplace=True)
            not_merged_mask = merged[target_columns[0]].isna()
            filled_rows = (~not_merged_mask) & (merged["LOD"]==-1)
            merged.loc[filled_rows, "LOD"] = lod
            not_merged = not_merged_mask.sum()
            df = merged
            if not_merged == 0:
                break
            else:
                #print(f"Could not merge {columns_to_merge_on} with {target_columns} on {not_merged} rows.")
                start = False
        if not_merged > 0:
            default_means:pd.Series = mean_encoding_dict["DefaultMean"]
            #add means to names in index
            default_means.index = [c+"Mean" for c in default_means.index]
            default_stds = mean_encoding_dict["DefaultStd"]
            default_stds.index = [c+"Std" for c in default_means.index]
            mean_cols = [c for c in default_means.index if c in df.columns]
            df[mean_cols].fillna(default_means, inplace=True)
            std_cols = [c for c in default_stds.index if c in df.columns]
            df[std_cols].fillna(default_stds, inplace=True)
            df.loc[df["LOD"]==-1, "LOD"] = 0
        dfs[i] = df
    return pd.concat(dfs, axis=0)

def enrich_traindf_with_mean_encodings(data: pd.DataFrame, mean_encoding_dict:Dict[str, Tuple[pd.DataFrame, int, int]]) -> pd.DataFrame:
    to_enrich_with_final = data.sample(frac=0.3)
    to_enrich_with_random = data.loc[data.index.difference(to_enrich_with_final.index)]
    to_enrich_with_final_2 = enrich_with_mean_encodings(to_enrich_with_final, mean_encoding_dict)
    to_enrich_with_random_2 = enrich_with_mean_encodings_random(to_enrich_with_random, mean_encoding_dict)
    #merge together the two enriched dataframes
    enriched_data = pd.concat([to_enrich_with_final_2, to_enrich_with_random_2])
    return enriched_data


if __name__ == '__main__':
    data = pd.read_csv("data/new_minutes_durations_history.csv")
    train_df, eval_df = stratify_data(data)
    mean_encoding_dict = mean_encoding(train_df)
    train_df_enriched = enrich_traindf_with_mean_encodings(train_df, mean_encoding_dict)
    eval_df_enriched = enrich_traindf_with_mean_encodings(eval_df, mean_encoding_dict)
    #train_df_enriched.to_csv("data/new_minutes_durations_history_mean_enriched_train.csv", index=False)
    #eval_df_enriched.to_csv("data/new_minutes_durations_history_mean_enriched_eval.csv", index=False)
    joblib.dump(mean_encoding_dict, "data/mean_encoding_dict.pkl")