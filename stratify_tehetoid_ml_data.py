from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Tuple

def stratify_data(data:pd.DataFrame, test_size=0.2, random_state=2142567) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratify the data based on the target variable."""
    return train_test_split(data, test_size=test_size, random_state=random_state%(2**32 - 1))


if __name__ == '__main__':
    data = pd.read_csv("data/TehETOID_for_machine_learning.csv")
    train_df, eval_df = stratify_data(data)
    train_df.to_csv("data/TehETOID_for_machine_learning_train.csv", index=False)
    eval_df.to_csv("data/TehETOID_for_machine_learning_eval.csv", index=False)