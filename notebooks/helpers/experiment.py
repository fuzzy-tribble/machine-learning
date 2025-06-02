import os
import pandas as pd
import joblib

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from helpers.const import RES_DIR, CACHE_DIR, FIGS_DIR
classification_exps_fname = "classification-experiments.csv"
regression_exps_fname = "regression-experiments.csv"
exps_cols = [
                "exp_name",
                "dataset_name",
                "n_train_samples",
                "n_test_samples",
                "mean_accuracy",
                "train_time",
                "query_time",
                "kfolds",
                "confusion_matrix",
                "classification_report",
            ]

def get_experiments_df(type="c"):
    """
    Load existing experiments df or create a new one
    """
    if type == "c": # classification
        if os.path.exists(f"{RES_DIR}/{classification_exps_fname}"):
            print(f"Loading '{classification_exps_fname}'")
            return pd.read_csv(f"{RES_DIR}/{classification_exps_fname}").set_index("exp_name")
        else:
            print(f"Creating experiments df: '{classification_exps_fname}'")
            
            exps = pd.DataFrame(columns=exps_cols).set_index("exp_name")
            exps.to_csv(f"{RES_DIR}/{classification_exps_fname}")
            return exps
    elif type == "r": # regression
        raise NotImplementedError("Regression experiments not implemented yet")
    else:
        raise ValueError(f"Unknown experiment type: {type}")

def get_experiment(type, name, create_new=False):
    """
    Load existing experiment or create a new one
    """
    exps_df = get_experiments_df(type)
    if name in exps_df.index:
        print(f"Loading '{name}' experiment")
        return exps_df.loc[[name]]
    else:
        if create_new:
            print(f"Creating experiment: '{name}'")
            df = pd.DataFrame(columns=exps_cols).set_index("exp_name")
            return df
        else:
            print(f"Experiment {name} not found. Use run get_experiment with create_new=True to create a new experiment")

def get_estimator(name, steps=[]):
    """
    Load existing persisted model/estimator (or pipeline) structure or create a new one based on provided steps
    """
    if os.path.exists(f"{CACHE_DIR}/{name}.joblib"):
        print(f"Loading '{name}' estimator/model/pipeline")
        return joblib.load(f"{CACHE_DIR}/{name}.joblib")
    else:
        print(f"No estimator found for '{name}'. Create a new one with memory={CACHE_DIR}")
        # for some reason creating pipeline with empty steps then adding them later causes issues.
        # return Pipeline(
        #     steps=steps, # may be empty, thats fine can be added later
        #     memory=CACHE_DIR,
        # )
        return None

class Experiment:
    def __init__(self, type, name, dataset):
        # type should be c or r only
        if type not in ["c", "r"]:
            raise ValueError("Experiment type should be 'c' for classification or 'r' for regression")
        self.type = type
        self.name = name # eg. dtc-unpruned, dtc-gridbest, etc.
        self.dataset = dataset # eg. iris-15test-shuffled-v1
        self._estimator: BaseEstimator = None

        # create cache and figs directories if they don't exist
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        if not os.path.exists(FIGS_DIR):
            os.makedirs(FIGS_DIR)
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)

        # load existing or create new experiment + estimator
        self.summary_df = get_experiment(type, name, create_new=True)
        self.update_param("dataset_name", dataset)
        
        est = get_estimator(name)
        if est is not None:
            self.estimator = est

    @property
    def estimator(self):
        return self._estimator
    
    @estimator.setter
    def estimator(self, new_estimator):
        self._estimator = new_estimator
        self._on_estimator_change()

    def _on_estimator_change(self):
        """
        When estimator object changes, save it to cache
        """
        joblib.dump(self.estimator, f"{CACHE_DIR}/{self.name}.joblib")
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.type}, {self.name}, {self.dataset})"

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
                    print(f"Parameter '{param_name}' already has value {val} in experiment summary. Use overwrite_existing=True to update. Skipping")

        # row exists but column does not, add the column and update the parameter value
        else:
            if add_column:
                print(f"Adding column: {param_name}")
                self.summary_df[param_name] = pd.NA
                self.summary_df.loc[self.name, param_name] = param_value
            else:
                raise ValueError(f"Parameter '{param_name}' not found in experiment summary")

    def save(self, overwrite_existing=False):
        """
        Save the experiment summary data
        """        
        exps = get_experiments_df("c")
        if self.name in exps.index:
            if not overwrite_existing:
                raise ValueError(f"Experiment {self.name} already exists. Use overwrite_existing=True to overwrite")
            else:
                print(f"Overwriting existing experiment {self.name}")
                exps = exps.drop(self.name)
        exps = pd.concat([exps, self.summary_df], axis=0)

        print(f"Saving experiment {self.name} to {RES_DIR}/{classification_exps_fname}")
        exps.to_csv(f"{RES_DIR}/{classification_exps_fname}")

        print(f"Dumping estimator {self.name} to {CACHE_DIR}/{self.name}.joblib")
        joblib.dump(self.estimator, f"{CACHE_DIR}/{self.name}.joblib")