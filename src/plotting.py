import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import src.data_preparation as dp
from sklearn.utils import shuffle
from fp_builds.filling_strategies import filling
from fp_builds import make_graph as mg
from fp_builds import data_loading, utils

import os


# divide the data into k folds, run the neural network on each fold and return the cross validation accuracy.
def k_fold(data: pd.DataFrame, func, k=5):
    """takes a dataframe and a number of folds and returns the cross validation accuracy of a neural network with 3
    hidden layers, trained on 80% of the data and tested on the remaining 20%."""

    # initialize the cross validation accuracy.
    c1, c2 = 0, 0
    # divide the data into k folds.

    # for each fold, train a neural network on the remaining folds and evaluate it on the current fold.
    for i in range(0, k):
        train, test = dp.split(data.copy(), 0.8, 0.2)

        # run the classifier on the training and test set.
        a, b = func(train, test)
        c1 += a
        c2 += b

    # return the cross validation accuracy.
    return c1 / k, c2 / k


# plots the AUC of the neural network as a function of the percentage of missing values, using the k-fold method with
# k=5. gets as input the original full dataset, an array of percentages of missing values, and the number of folds.
def auc(data_name: str, percentages: np.array, func, name, k=5):
    """takes a dataframe, an array of percentages of missing values and a number of folds and plots the AUC of the
    neural network over the test data as a function of the percentage of missing values, using the k-fold method with
    k=5."""

    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    # initialize the AUC array.
    train_auc = np.zeros(len(percentages))
    test_auc = np.zeros(len(percentages))
    # prints what function is used.
    print(f"function: {name}")
    # for each percentage of missing values, run the neural network using the k-fold method and calculate the AUC.
    for i in range(0, len(percentages)):
        print(f"missing percentage: {int(percentages[i] * 100)}%")
        # run the program 10 times and take avg.

        # if the data is graph.
        if data_name in ["Cora", "Citeseer", "Pubmed"]:
            dataset, _ = data_loading.get_dataset(data_name)
            data = dataset.data
            n_nodes, n_features = data.x.shape

            missing_feature_mask = utils.get_missing_feature_mask(
                rate=i / 10, n_nodes=n_nodes, n_features=n_features, type="uniform",
            )
            x = data.x.clone()
            x[~missing_feature_mask] = np.nan

            y = data.y.clone()

            data_missing = pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)

            a, b = k_fold(data_missing, func, k)
            train_auc[i] += a
            test_auc[i] += b
            continue
        # initialize the data with missing values.
        path = os.path.dirname(__file__)
        data_ = pd.read_csv(f"{path}/Datasets/{data_name}.csv")
        data = data_.copy()

        # shuffle.
        data = shuffle(data)

        data_missing, _ = dp.remove_random_cells_and_z_score(data, percentages[i])
        # initialize the cross validation accuracy.
        a, b = k_fold(data_missing, func, k)
        train_auc[i] += a
        test_auc[i] += b

    # returns the AUC array.
    return train_auc, test_auc


# def auc_opt(data: pd.DataFrame, percentages: np.array, func, name):
#     # initialize the AUC array.
#     train_auc = np.zeros(len(percentages))
#     test_auc = np.zeros(len(percentages))
#     #prints what function is used.
#     print(f"function: {name}")
#     # for each percentage of missing values, run the neural network using the k-fold method and calculate the AUC.
#     for i in range(0, len(percentages)):
#         print(f"missing percentage: {int(percentages[i] * 100)}%")
#
#         # run 10 times and take avg.
#         for j in range(10):
#
#             # initialize the data with missing values.
#             data_missing = dp.remove_random_cells(data.copy(), percentages[i])
#             train, val, test = dp.split(data_missing, 0.64, 0.16, 0.2)
#
#             a1, a2 = func(train, val, test)
#             train_auc[i] += a1
#             test_auc[i] += a2
#
#         train_auc[i] /= 10
#         test_auc[i] /= 10
#
#     return train_auc, test_auc


# method to plot the AUC as a function of the percentage of missing values, using the k-fold method with k=5,
# with data imputation.
def auc_imputation(data_name: str, percentages: np.array, func, name, graph_style: str = "knn", k=5):
    """takes a dataframe, an array of percentages of missing values and a number of folds and plots the AUC of the
    neural network over the test data as a function of the percentage of missing values, using the k-fold method with
    k=5."""

    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    # initialize the AUC array.
    train_auc = np.zeros(len(percentages))
    test_auc = np.zeros(len(percentages))
    # prints what function is used.
    print(f"function: {name} with imputation")
    # for each percentage of missing values, run the neural network using the k-fold method and calculate the AUC.
    for i in range(0, len(percentages)):
        print(f"missing percentage: {int(percentages[i] * 100)}%")

        # run the program 10 times and take avg.

        # if the data is a graph.
        if data_name in ["Cora", "Citeseer", "Pubmed"]:
            dataset, _ = data_loading.get_dataset(data_name)
            data = dataset.data
            n_nodes, n_features = data.x.shape

            missing_feature_mask = utils.get_missing_feature_mask(
                rate=i / 10, n_nodes=n_nodes, n_features=n_features, type="uniform",
            ).to(device)
            x = data.x.clone()
            x[~missing_feature_mask] = np.nan

            y = data.y.clone()

            filled_features = (
                filling("feature_propagation", data.edge_index, x, missing_feature_mask, 40, )
            )

            x = filled_features
            data_filled = pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)

            a, b = k_fold(data_filled, func, k)
            train_auc[i] += a
            test_auc[i] += b
            continue

        # initialize the data with missing values.
        path = os.path.dirname(__file__)
        data_ = pd.read_csv(f"{path}/Datasets/{data_name}.csv")
        data = data_.copy()

        # shuffle.
        data = shuffle(data)

        data_missing, mask = dp.remove_random_cells_and_z_score(data, percentages[i])

        # impute the missing values.
        x = data_missing.drop(data_missing.columns[-1], axis=1)
        y = torch.from_numpy(data_missing.iloc[:, -1].values.astype(np.int64))

        # checking nan entries.
        z = pd.DataFrame(x.copy().notna().astype(int))
        z = pd.concat([x, z], axis=1)

        # creating edges.

        if graph_style == "knn":  # knn graph with heuristic distance and default k=3.
            edges, distances = mg.make_graph_knn(z)
            d = mg.find_mean_dist(x, edges)
            print(f"mean distance for method: {graph_style}, with missing rate {(i+1) * 10}% is: {np.mean(d):.3f}"
                  f"+-{np.std(d):.3f}")

        elif graph_style == "random":  # each node is connected to 3 other nodes.
            edges = mg.make_graph_random(x)
            d = mg.find_mean_dist(x, edges)
            print(f"mean distance for method: {graph_style}, with missing rate {(i+1) * 10}% is: {np.mean(d):.3f}"
                  f"+-{np.std(d):.3f}")

        elif graph_style == "forward":  # each node is connected to the next 3 nodes.
            edges = mg.make_graph_forward(x)
            d = mg.find_mean_dist(x, edges)
            print(f"mean distance for method: {graph_style}, with missing rate {(i+1) * 10}% is: {np.mean(d):.3f}"
                  f"+-{np.std(d):.3f}")

        elif graph_style == "backward":  # each node is connected to the previous 2 nodes and next node.
            edges = mg.make_graph_backward(x)
            d = mg.find_mean_dist(x, edges)
            print(f"mean distance for method: {graph_style}, with missing rate {(i+1) * 10}% is: {np.mean(d):.3f}"
                  f"+-{np.std(d):.3f}")

        x = torch.from_numpy(x.values.astype(np.float32)).to(device)

        filled_features = (
            filling("feature_propagation", edges, x, mask, 40, )
        )

        x = filled_features
        data_imputed = pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)
        # initialize the cross validation accuracy.
        a, b = k_fold(data_imputed, func, k)
        train_auc[i] += a
        test_auc[i] += b

    return train_auc, test_auc


# method to plot the AUC as a function of the percentage of missing values, using optimization method and imputing
# missing data.
# def auc_opt_imputation(data: pd.DataFrame, percentages: np.array, func, name):
#     # initialize the AUC array.
#     train_auc = np.zeros(len(percentages))
#     test_auc = np.zeros(len(percentages))
#     # prints what function is used.
#     print(f"function: {name}")
#     # for each percentage of missing values, run the neural network using the k-fold method and calculate the AUC.
#     for i in range(0, len(percentages)):
#         print(f"missing percentage: {int(percentages[i] * 100)}%")
#
#         # run 10 times and take avg.
#         for j in range(10):
#
#             # initialize the data with missing values.
#             data_missing = dp.remove_random_cells(data.copy(), percentages[i], True)
#             data_missing = build_fp.impute(data_missing)
#             train, val, test = dp.split(data_missing, 0.64, 0.16, 0.2)
#             a, b = func(train, val, test)
#             train_auc[i] += a
#             test_auc[i] += b
#
#         train_auc[i] /= 10
#         test_auc[i] /= 10
#
#     return train_auc, test_auc


def plot_multiple_graphs(x_values, *args, labels=None, title="Multiple Graphs", x_label="% of missing values",
                         y_label="AUC score", legend_loc="upper right", data_name="", method_name="xgb"):
    """
    Plot multiple graphs on the same plot.

    Parameters:
    - x_values: X-axis values.
    - *args: Variable number of lists of values to be plotted.
    - labels (list): List of labels for each set of values.
    - title (str): Title of the plot.
    - x_label (str): Label for the X-axis.
    - y_label (str): Label for the Y-axis.
    - legend_loc (str): Location of the legend.

    Returns:
    - None (displays the plot).
    """

    if labels is None:
        labels = [f"Plot {i + 1}" for i in range(len(args))]

    if len(args) != len(labels):
        raise ValueError("Number of arguments and number of labels must be the same.")

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, args[0], label=labels[0], marker="o")

    for i in range(1, len(args)):
        plt.plot(x_values, args[i], label=labels[i], linestyle="-", marker="o")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)
    plt.grid(True)

    # Create a "plots" directory if it doesn't exist
    save_folder = "plots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the figure
    file_name = f"{data_name}_{method_name}_fig.png"
    plt.savefig(os.path.join(save_folder, file_name))
    print(f"saved at {os.path.join(save_folder, file_name)}")
    plt.show()
