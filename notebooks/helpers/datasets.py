import os
import pandas as pd

from helpers.const import DATA_DIR, RES_DIR

edas_fname = "edas.csv"
edas_cols = [
                "dataset",
                "description",
                "n features",
                "n samples",
                "f/n ratio", # are there enough samples for the number of features?
                "noise", # are there noisy features like missing values or ones with same target classes? some models are more/less sensitive to noise
                "stats", # how much variance is there in the data?
                "class balance", # are target classes balanced?
                "outliers", # are there outliers?
                "skewness",  # are features normally distributed? If not, it may affect KNN, SVC, etc.
                "correlations", # are there high correlations between features? if so, they may be redundant
                "DR potential"  # is there any dimensionality reduction potential?
            ]

def get_dataset(dataset_name):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # try load data from dataset name regex
    try:
        notes = open(f"{DATA_DIR}/{dataset_name}-notes.txt").read()
        X_train = pd.read_csv(f"{DATA_DIR}/{dataset_name}-Xtrain.csv")
        X_test = pd.read_csv(f"{DATA_DIR}/{dataset_name}-Xtest.csv")
        y_train = pd.read_csv(f"{DATA_DIR}/{dataset_name}-ytrain.csv")
        y_test = pd.read_csv(f"{DATA_DIR}/{dataset_name}-ytest.csv")
        try:
            target_names = pd.read_csv(f"{DATA_DIR}/{dataset_name}-targetnames.csv")
        except FileNotFoundError as e:
            target_names = None
    except FileNotFoundError as e:
        raise ValueError(f"Could not load data for dataset {dataset_name}. Please ensure data is available in {DATA_DIR} directory")
    return notes, X_train, X_test, y_train, y_test, target_names

def save_dataset(dataset_name, X_train, X_test, y_train, y_test, target_names=None, dataset_notes=None, overwrite_existing=False):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(f"{DATA_DIR}/{dataset_name}-*") and not overwrite_existing:
            raise ValueError(f"Dataset {dataset_name} already exists. Use overwrite_existing=True to overwrite")

    X_train.to_csv(f"{DATA_DIR}/{dataset_name}-Xtrain.csv", index=False)
    X_test.to_csv(f"{DATA_DIR}/{dataset_name}-Xtest.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/{dataset_name}-ytrain.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/{dataset_name}-ytest.csv", index=False)
    if target_names is not None:
        target_names.to_csv(f"{DATA_DIR}/{dataset_name}-targetnames.csv", index=False)
    # save dataset notes/metadata to file
    with open(f"{DATA_DIR}/{dataset_name}-notes.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_test shape: {X_test.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"y_test shape: {y_test.shape}\n")
        f.write(f"Train: {X_train.shape[0]/(X_train.shape[0] + X_test.shape[0]) * 100:.2f}% of total\n")
        f.write(f"Test: {X_test.shape[0]/(X_train.shape[0] + X_test.shape[0]) * 100:.2f}% of total\n")
        f.write(f"Notes: {dataset_notes}\n")
        f.write(f"Created by save_dataset() helper at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def get_edas_df():
    """
    Load existing df or create a new one
    """
    if os.path.exists(f"{RES_DIR}/{edas_fname}"):
        print(f"Loading '{edas_fname}'")
        return pd.read_csv(f"{RES_DIR}/{edas_fname}").set_index("dataset")
    else:
        print(f"Creating edas df: '{edas_fname}'")
        
        exps = pd.DataFrame(columns=edas_cols).set_index("dataset")
        exps.to_csv(f"{RES_DIR}/{edas_fname}")
        return exps

def get_eda(name, create_new=False):
    """
    Load existing or create a new one
    """
    exps_df = get_edas_df()
    if name in exps_df.index:
        if create_new:
            print(f"Can't create a new eda if one with the same name already exists: '{name}'")
        else:
            print(f"Loading '{name}' eda")
            return exps_df.loc[[name]]
    else:
        if create_new:
            print(f"Creating experiment: '{name}'")
            df = pd.DataFrame(columns=edas_cols).set_index("dataset")
            return df
        else:
            print(f"EDA {name} not found. Use run get_experiment with create_new=True to create a new")
class EDA:
    def __init__(self, name):
        self.name = name
        
        # create directories if they don't exist
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)

        # load existing or create new
        self.summary_df = get_eda(name)

    def update_param(self, param_name, param_value, overwrite_existing=True, add_column=False):
        """
        Update the experiment summary with a new parameter value
        """
        
        # row does not exist, create it and add the parameter
        if self.name not in self.summary_df.index:
                self.summary_df.loc[self.name, param_name] = param_value
        
        # row and column exists, update parameter value
        elif param_name in self.summary_df.columns:
                val = self.summary_df.loc[self.name, param_name]
                if pd.isna(val) or overwrite_existing:
                    self.summary_df.loc[self.name, param_name] = param_value
                else:
                    print(f"Parameter '{param_name}' already has value {val} in summary. Use overwrite_existing=True to update. Skipping")

        # row exists but column does not, add the column and update the parameter value
        else:
            if add_column:
                print(f"Adding column: {param_name}")
                self.summary_df[param_name] = pd.NA
                self.summary_df.loc[self.name, param_name] = param_value
            else:
                raise ValueError(f"Parameter '{param_name}' not found in summary")

    def __repr__(self):
        # print column names and dataset names
        if self.summary_df is None:
            return f"EDA: {self.name}\nNo summary data available"
        return f"EDA: {self.name}\nColumns: {self.summary_df.columns}\nDatasets: {self.summary_df.index}"

    def save(self, overwrite_existing=False):
        """
        Save the summary data
        """        
        exps = get_edas_df()
        if self.name in exps.index:
            if not overwrite_existing:
                raise ValueError(f"{self.name} already exists. Use overwrite_existing=True to overwrite")
            else:
                print(f"Overwriting existing {self.name}")
                exps = exps.drop(self.name)
        exps = pd.concat([exps, self.summary_df], axis=0)

        print(f"Saving {self.name} to {RES_DIR}/{edas_fname}")
        exps.to_csv(f"{RES_DIR}/{edas_fname}")