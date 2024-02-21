from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


# insert data into dataframe. returns the data without the first column (indexing column)
def csv_to_df(file_directory: str) -> pd.DataFrame:
    """takes a directory and returns a dataframe."""

    frame = pd.read_csv(file_directory, sep=",")
    labels = frame.iloc[:, -1]
    frame = frame.drop(frame.columns[-1], axis=1)
    # paste the labels back on the dataframe.
    frame = pd.concat([frame, labels], axis=1)
    return frame


# insert data into dataframe. gets input as pickle file.
def pkl_to_df(file_directory: str) -> pd.DataFrame:
    """takes a directory and returns a dataframe."""
    frame = pd.read_pickle(file_directory)
    print(frame)
    labels = frame.iloc[:, -1]
    frame = frame.drop(frame.columns[-1], axis=1)
    # paste the labels back on the dataframe.
    frame = pd.concat([frame, labels], axis=1)
    return frame


# z-score each column.

def z_score(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and z-scores each column respectively.

    Parameters:
    - data_frame: Input DataFrame

    Returns:
    - z_scored_frame: DataFrame with z-scored columns
    """
    for i in range(data_frame.shape[1] - 1):
        column_mean = data_frame.iloc[:, i].mean(skipna=True)
        column_std = data_frame.iloc[:, i].std(skipna=True)

        if column_std != 0 and not np.isnan(column_std):  # Avoid division by zero when std is zero
            data_frame.iloc[:, i] = (data_frame.iloc[:, i] - column_mean) / column_std

    return data_frame


# remove percentage of random cells by replacing them with 0.
def remove_random_cells(data_frame: pd.DataFrame, percentage: float):
    """takes a dataframe and a percentage and replaces that percentage of cells with 0.
    the cells are chosen randomly."""
    mask = np.ones(data_frame.iloc[:, : -1].shape)
    for i in range(0, data_frame.shape[0]):
        for j in range(0, data_frame.shape[1] - 1):
            if np.random.rand() < percentage:
                mask[i, j] = 0
                data_frame.iloc[i, j] = float("nan")

    # convert mask to tensor.
    mask = torch.from_numpy(mask).bool()

    return data_frame, mask


# takes input of a DataFrame, and return it split into training, testing, and validation (defaulted to 0%).
def split(frame: pd.DataFrame, train: float, test: float, val: float = 0):
    target = frame.iloc[:, -1]
    features = frame.iloc[:, :-1]
    # splitting the set (first checking if val != 0).
    if val != 0:
        # dummy size is val + test.
        x_train, x_dummy, y_train, y_dummy = train_test_split(features, target, train_size=train)

        # initialize test set.
        x_val, x_test, y_val, y_test = train_test_split(x_dummy, y_dummy, test_size=test / (val + test))

        # concat the x_,y_ arrays into one DataFrame for the train, val and test sets.
        train_set = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1)
        val_set = pd.concat([pd.DataFrame(x_val), pd.DataFrame(y_val)], axis=1)
        test_set = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
        return train_set, val_set, test_set

    # if val is 0.
    x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=train)
    train_set = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1)
    test_set = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    return train_set, test_set


# check if there are duplicated rows in the data.
def check_duplicates(data: pd.DataFrame):
    """takes a dataframe and checks if there are duplicated rows in the data, if so, removes them."""
    # check if there are duplicated rows in the data.
    if data.duplicated().any():
        # remove duplicated rows.
        print(data.duplicated().any())
        data.drop_duplicates(inplace=True)
    return data
