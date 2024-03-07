
import pandas as pd
from duration_in_minutes.data_utils import normalize_product_column, normalize_operation_column
from supply_prediciton.data_utils import get_embedding_column_for

file_path = "data/duration_in_minutes_history.csv"
data = pd.read_csv(file_path)

normalised = normalize_product_column(data[["FinalProductName"]], "FinalProductName")
data = pd.concat([data, normalised], axis=1)

operation_normalised = normalize_operation_column(data[["OperationDescription"]], "OperationDescription")
data = pd.concat([data, operation_normalised], axis=1).drop_duplicates()

resource_normalised = normalize_product_column(data[["ResourceName"]], "ResourceName")
data = pd.concat([data, resource_normalised], axis=1)

embeddings = get_embedding_column_for(data[["ResourceNameNorm", "OperationDescriptionNorm", "ResourceNameNorm"]])
data = pd.concat([data, embeddings], axis=1)

data.to_csv("data/duration_in_minutes_history_enrich.csv", index=False)