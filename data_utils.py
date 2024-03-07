from typing import Dict, Tuple
import pandas as pd
import re
from datetime import datetime, date
#from sentence_transformers import SentenceTransformer

def find_product_name_in_row(value:str, code_regex:re.Pattern):
    #res = code_regex.findall(value)
    match = code_regex.search(value)
    if match is None:
        return -1
    if len(match.regs) == 0:
        return -1
    return match.regs[0][0]

def get_month(df: pd.DataFrame):
    colname = df.columns[0]
    new_df = pd.DataFrame({"Month": df[[colname]].apply(
        lambda row: row[colname].split("-")[1] if isinstance(row[colname], str) else "Unknown", axis=1)
    })
    return new_df

def get_weekday(df: pd.DataFrame):
    colname = df.columns[0]
    #parse weekday from date column
    df[colname] = pd.to_datetime(df[colname], errors='raise')
    new_df = pd.DataFrame({"Weekday": df[[colname]].apply(
        lambda row: row[colname].weekday(), axis=1)
    })
    return new_df

def calculate_group_means_2(df: pd.DataFrame, target_column_name="DurationInMinutes"):
    df = df.copy()
    df['JOB_CAT'] = df[['ItemType','ItemGroup','ItemCode','ResourceGroup','StandardOperationCode']].apply(lambda r : f"{r['ItemType']}-{r['ItemGroup']}-{r['ItemCode']}-{r['ResourceGroup']}-{r['StandardOperationCode']}", axis=1)
    df_groups = df.groupby("JOB_CAT")[target_column_name].mean()
    column = df_groups.loc[df["JOB_CAT"]].reset_index()[[target_column_name]]
    column.rename(columns={target_column_name:target_column_name+'_NORM'}, inplace=True)
    return column

def calculate_group_means(df: pd.DataFrame, target_column_name="DurationInMinutes"):
    df = df.copy()
    df['JOB_CAT'] = df[["ResourceNameNorm", "OperationDescriptionNorm", "FinalProductNameNorm"]].apply(lambda r : f"{r['ResourceNameNorm']}-{r['OperationDescriptionNorm']}-{r['FinalProductNameNorm']}", axis=1)
    df_groups = df.groupby("JOB_CAT")[target_column_name].mean()
    column = df_groups.loc[df["JOB_CAT"]].reset_index()[[target_column_name]]
    column.rename(columns={target_column_name:target_column_name+'_NORM'}, inplace=True)
    return column

def integrate_group_means(data: pd.DataFrame, group_means: pd.DataFrame):
    group_mean_cols = [c for c in group_means.columns if c.endswith("_NORM")]
    group_means = group_means.groupby(['ItemType','ItemGroup','ItemCode','ResourceGroup','StandardOperationCode']).mean().reset_index()
    merged_data = data.merge(group_means, on=['ItemType','ItemGroup','ItemCode','ResourceGroup','StandardOperationCode'], how="left")
    all_means = group_means[group_mean_cols].mean(axis=0)
    merged_data.fillna(all_means, inplace=True)
    return merged_data[group_mean_cols]
    

def normalize_product_column(data: pd.DataFrame, column_name="ItemGroup"):
    code_regex = re.compile(r"[ A-Z0-9-_/.,']{4,}")
    regex_end = re.compile(r"[ A-Z0-9-/.,']{4,}.*")
    regex_start = re.compile(r"[ A-Z0-9-/,.']{4,}")
    new_df = pd.DataFrame({(column_name+"Norm"): data[[column_name]].apply(
        lambda row:
            regex_start.sub(" ", row[column_name]).strip() if find_product_name_in_row(row[column_name].strip(), code_regex) < max(0.25*len(row[column_name].strip()), 2) else regex_end.sub(" ", row[column_name]).strip()
        ,axis=1)
    })
    new_df[new_df[column_name+'Norm'] == ''] = 'Nepoznato'
    return new_df

def modify_operation_value(value:str, code_regex, regex_end, regex_start):
    value = value.split("-")[0].strip()
    return regex_start.sub(" ", value).strip() if find_product_name_in_row(value, code_regex) < max(0.25*len(value), 2) else regex_end.sub(" ", value).strip()


def normalize_operation_column(data: pd.DataFrame, column_name="ItemGroup"):
    code_regex = re.compile(r"[ A-Z0-9-_/.,']{4,}")
    regex_end = re.compile(r"[ A-Z0-9-/.,']{4,}.*")
    regex_start = re.compile(r"[ A-Z0-9-/,.']{4,}")
    new_df = pd.DataFrame({(column_name+"Norm"): data[[column_name]].apply(
        lambda row:
            modify_operation_value(str(row[column_name]).strip(), code_regex, regex_end, regex_start)
        ,axis=1)
    })
    new_df[new_df[column_name+'Norm'] == ''] = 'Nepoznato'
    return new_df

def add_means_from_means_encoding(data: pd.DataFrame, mean_encoding_dict:Dict[str, Tuple[pd.DataFrame, int, int]]):
    new_df = data.copy()

    for combination, (means, _, _) in mean_encoding_dict.items():
        if combination == "DefaultMean":
            continue
        new_df = new_df.merge(means, on=list(combination), how="left")
    return new_df
    

# def get_embedding_column_for(data: pd.DataFrame):
#     sentences = data.apply(lambda row: ' - '.join(row.to_list()), axis=1)
#     inverted_index = defaultdict(list)
#     for i, sentence in enumerate(sentences):
#         lst = inverted_index[sentence.lower().strip()]
#         lst.append(i)
    
#     sentences_unique = list(inverted_index.keys())

#     c = Counter()
#     for s in sentences_unique:
#         c.update(s)
#     total_count = sum(c.values())
#     allowed_chars = set([char for char, count in c.items() if count >= 0.001*total_count])

#     embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
#     embeddings_arr = np.zeros((len(data), 512), dtype=np.float32)

#     for sentence, indices in tqdm(inverted_index.items()):
#         sentence = ''.join([c for c in sentence if c in allowed_chars])
#         embs = embedder.encode([sentence])
#         for i in indices:
#             embeddings_arr[i] = embs[0]
    
#     column_dict = {}
#     for i in range(512):
#         #column_dict["NAME_EMB_" + str(i)] = embeddings_name[:,i]
#         column_dict["GROUP_EMB_" + str(i)] = embeddings_arr[:,i]
    
#     return pd.DataFrame(column_dict)

"""
def get_separate_embeddings_columns(data: pd.DataFrame, dictionary:Dict=None):
    assert dictionary is not None
    for v in dictionary.values():
        emb = v
        break
    len_emb = len(emb)
    len_data = len(data)

    #embeddings_name = np.zeros((len_data, len_emb), dtype=np.float32)
    embeddings_group = np.zeros((len_data, len_emb), dtype=np.float32)

    for id in data["Unnamed: 0"]:
        name_emb = dictionary[(id, "ItemName")]
        group_emb = dictionary[(id, "ItemGroup")]
        #embeddings_name[id] = name_emb
        embeddings_group[id] = group_emb

    column_dict = {}
    for i in range(len_emb):
        #column_dict["NAME_EMB_" + str(i)] = embeddings_name[:,i]
        column_dict["GROUP_EMB_" + str(i)] = embeddings_group[:,i]

    return pd.DataFrame(column_dict)

def get_sentence_embeddings_column(data: pd.DataFrame, dictionary:Dict=None):
    assert dictionary is not None
    for v in dictionary.values():
        emb = v
        break
    len_emb = len(emb)
    len_data = len(data)

    embeddings_all = np.zeros((len_data, len_emb), dtype=np.float32)

    for id in data["Unnamed: 0"]:
        name_emb = dictionary[id]
        embeddings_all[id] = name_emb

    column_dict = {}
    for i in range(len_emb):
        column_dict["ALL_EMB_" + str(i)] = embeddings_all[:,i]

    return pd.DataFrame(column_dict)

"""