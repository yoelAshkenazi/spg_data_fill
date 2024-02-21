import pandas as pd
import numpy as np
import os
from src import build_classifier as classify
from sklearn.model_selection import train_test_split
from fp_builds import data_loading, utils
import data_preparation as dp
from fp_builds.filling_strategies import filling
from fp_builds import make_graph as mg
# from sklearn.utils import shuffle
import torch
from src.data_filler import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def test_method(name: str, method_name: str, k: int) -> (list, list):
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    # set up dictionary of metric constants.
    constants = {
        0.1: [0.839, 0, 1.366, 1.109],
        0.2: [0.882, 0, 1.2, 1.265],
        0.3: [0.963, 0, 1.074, 1.14],
        0.4: [1.027, 0, 1.035, 1.035],
        0.5: [1.026, 0, 1.026, 1.032],
        0.6: [1.026, 0, 1.026, 1.026],
        0.7: [1.026, 0, 0.996, 1.024],
        0.8: [1.024, 0, 0.958, 1.024],
        0.9: [0.981, 0, 1.014, 1.036],
    }

    # results.
    a, b, c = [], [], []
    cheat = []
    # run the method.
    for i in range(1, 10):
        print(f"working on {name} with missing rate {i * 10}%")
        # run on unfilled data.
        if name in ["Cora", "Citeseer", "Pubmed"]:
            dataset, _ = data_loading.get_dataset(name)
            data = dataset.data
            n_nodes, n_features = data.x.shape

            missing_feature_mask = utils.get_missing_feature_mask(
                rate=i / 10, n_nodes=n_nodes, n_features=n_features, type="uniform",
            ).to(device)

            data_original = pd.concat([pd.DataFrame(data.x.cpu().detach().numpy()), pd.DataFrame(data.y)], axis=1)

            x = data.x.clone()
            x[~missing_feature_mask] = np.nan

            y = data.y.clone()

        # else: the dataset is tabular, hence we need to create edges.
        else:
            data_ = pd.read_csv(f"data/{name}.csv")

            # shuffle.
            data_ = shuffle(data_)

            data_ = dp.z_score(data_)
            x_ = data_.iloc[:, :-1].copy()
            # normalize.
            data = dp.z_score(data_.copy())
            data, mask = dp.remove_random_cells(data, i / 10)

            y = data.iloc[:, -1].copy()
            x = data.drop(data.columns[-1], axis=1)

            x = torch.from_numpy(x.values.astype(np.float32))

            data_original = pd.concat([x_, pd.DataFrame(y)], axis=1)

            y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64))

        x = pd.DataFrame(x.cpu().detach().numpy())
        data_unfilled = pd.concat([x, pd.DataFrame(y)], axis=1)

        # constructing the missing features.
        if name in ["Cora", "Citeseer", "Pubmed"]:
            x = torch.from_numpy(x.values.astype(np.float32)).to(device)
            filled_features = (
                filling("feature_propagation", data.edge_index, x, missing_feature_mask, 40, )
            )
        else:
            # calculate masks.
            t_t, xor_vals, t_f, f_t, truth_mask = calc_masks(x)
            # calculate distances.
            dists = calc_l2_normal(x.shape[1], t_t, xor_vals, t_f, f_t, truth_mask, np.array(constants[i / 10]))
            edges = mg.get_knn_edges(dists, k)  # get edges.

            full_dists = calc_l2(x_)
            full_edges = mg.get_knn_edges(full_dists, k)

            x = torch.from_numpy(x.values.astype(np.float32)).to(device)

            filled_features = (
                filling("feature_propagation", edges, x, mask, 40, )
            )

            filled_features_cheat = (
                filling("feature_propagation", full_edges, x, mask, 40, )
            )

        data_filled = pd.concat([pd.DataFrame(filled_features.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)
        data_filled_cheat = pd.concat([pd.DataFrame(filled_features_cheat.cpu().detach().numpy()), pd.DataFrame(y)],
                                      axis=1)

        # split the data.
        og_train, og_test = train_test_split(data_original, test_size=0.2)
        filled_train, filled_test = train_test_split(data_filled, test_size=0.2)
        unfilled_train, unfilled_test = train_test_split(data_unfilled, test_size=0.2)
        cheat_train, cheat_test = train_test_split(data_filled_cheat, test_size=0.2)

        # if neural network, fill the nan with 0.
        if method_name == "nn":
            unfilled_train = unfilled_train.fillna(0)
            unfilled_test = unfilled_test.fillna(0)
            # run the nn.
            _, a_ = classify.run_nn(unfilled_train, unfilled_test)
            _, b_ = classify.run_nn(filled_train, filled_test)
            _, c_ = classify.run_nn(og_train, og_test)
            a.append(a_)
            b.append(b_)
            c.append(c_)
            continue

        # run on unfilled data.
        _, a_ = classify.run_xgb(unfilled_train, unfilled_test)
        # run on filled data.
        _, b_ = classify.run_xgb(filled_train, filled_test)
        # run on original.
        _, c_ = classify.run_xgb(og_train, og_test)

        # run on filled data with cheating.
        _, cheat_ = classify.run_xgb(cheat_train, cheat_test)

        a.append(a_)
        b.append(b_)
        c.append(c_)
        cheat.append(cheat_)

    return a, b, c, cheat


def test_filling(name: str, graph_style: str = "knn", k: int = 3, print_dist_data: bool = False) -> (list, list):
    """

    :param name: the name of the dataset.
    :param graph_style: the method of adding edges. default to knn.
    :return: the difference between the original data to the filled and unfilled data.
    """
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )
    print(f"working on {name}")

    # save results.
    diff_unfilled = np.zeros(9)
    diff_filled = np.zeros(9)

    for i in range(1, 10):
        for _ in range(5):
            # if graph already.
            if name in ["Cora", "Citeseer", "Pubmed"]:
                dataset, _ = data_loading.get_dataset(name)
                data = dataset.data
                n_nodes, n_features = data.x.shape

                missing_feature_mask = utils.get_missing_feature_mask(
                    rate=i / 10, n_nodes=n_nodes, n_features=n_features, type="uniform",
                ).to(device)
                x = data.x.clone()

                x = pd.DataFrame(x.cpu().detach().numpy())
                x = dp.z_score(x)

                x = torch.from_numpy(x.values.astype(np.float32)).to(device)
                x[~missing_feature_mask] = np.nan

                x = pd.DataFrame(x.cpu().detach().numpy())

                # impute.
                x = torch.from_numpy(x.values.astype(np.float32)).to(device)
                filled_features = (
                    filling("feature_propagation", data.edge_index, x, missing_feature_mask, 40, )
                )

                # return the differences.
                frame1 = data.x.cpu().detach().numpy()  # original data.
                frame2 = np.nan_to_num(filled_features.cpu().detach().numpy())  # filled data.
                frame3 = np.nan_to_num(x.cpu().detach().numpy())  # unfilled data.
                # normalize all frames.
                frame1 /= np.linalg.norm(frame1)
                frame2 /= np.linalg.norm(frame2)
                frame3 /= np.linalg.norm(frame3)

                # calculating the difference.
                n1 = np.linalg.norm(frame1 - frame2)
                n2 = np.linalg.norm(frame1 - frame3)

                diff_filled[i - 1] += n1

                diff_unfilled[i - 1] += n2

                continue

            # else: the dataset is tabular, hence we need to create edges.
            path = os.path.dirname(__file__)
            data_ = pd.read_csv(f"{path}/Datasets/{name}.csv")
            data = data_.copy()

            # shuffle.
            data = shuffle(data)
            # normalize.
            data = dp.z_score(data)
            data, mask = dp.remove_random_cells(data, i / 10)

            x = data.drop(data.columns[-1], axis=1)

            # creating edges.
            if graph_style == "knn":
                edges, distances = mg.make_graph_knn(x, k=k)
                d = mg.find_mean_dist(x, edges)

            elif graph_style == "random":
                edges = mg.make_graph_random(x)
                d = mg.find_mean_dist(x, edges)

            elif graph_style == "forward":
                edges = mg.make_graph_forward(x)
                d = mg.find_mean_dist(x, edges)

            if print_dist_data:
                print(f"mean distance for method: {graph_style}, with missing rate {i * 10}% is: {np.mean(d):.3f}"
                      f"+-{np.std(d):.3f}")

            # filling.
            x = torch.from_numpy(x.values.astype(np.float32)).to(device)
            filled_features = (
                filling("feature_propagation", edges, x, mask, 40, )
            )

            # return the differences.

            frame1 = data_.iloc[:, :-1].values  # original data.
            frame2 = np.nan_to_num(filled_features.cpu().detach().numpy())  # filled data.
            frame3 = np.nan_to_num(x.cpu().detach().numpy())  # unfilled data.
            # normalize all frames.
            frame1 /= np.linalg.norm(frame1)
            frame2 /= np.linalg.norm(frame2)
            frame3 /= np.linalg.norm(frame3)

            # calculating the difference.
            n1 = np.linalg.norm(frame1 - frame2)
            n2 = np.linalg.norm(frame1 - frame3)

            diff_filled[i - 1] += n1

            diff_unfilled[i - 1] += n2

        print("missing rate: ", i * 10, "%")
        print(f"the difference between the original data to the filled and unfilled data is: "
              f"{n1:.3f} and {n2:.3f} respectively.")

        diff_unfilled[i - 1] /= 10
        diff_filled[i - 1] /= 10

    return diff_unfilled, diff_filled


def test_iterations(name: str, graph_style: str = "knn", k: int = 3, print_dist_data: bool = False, ) -> (list, list):
    """

    :param name: the name of the dataset.
    :param graph_style: the method of adding edges. default to knn.
    :param k: used for knn.
    :param print_dist_data: flag deciding whether to print the mean distance data.
    :return: two lists of length 9, containing the difference between the original data to the filled and unfilled data
     as a function of the number of FP iterations.
    """
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )
    print(f"working on {name}")

    # save results.
    diff_unfilled = np.zeros(9)
    diff_filled = np.zeros(9)

    # loop over all missing rates.
    for i in range(1, 10):
        # if graph already.
        if name in ["Cora", "Citeseer", "Pubmed"]:
            dataset, _ = data_loading.get_dataset(name)
            data = dataset.data
            n_nodes, n_features = data.x.shape
            missing_feature_mask = utils.get_missing_feature_mask(
                rate=i / 10, n_nodes=n_nodes, n_features=n_features, type="uniform",
            ).to(device)
            x = data.x.clone()
            x = pd.DataFrame(x.cpu().detach().numpy())
            x = dp.z_score(x)
            x = torch.from_numpy(x.values.astype(np.float32)).to(device)
            x[~missing_feature_mask] = np.nan
            filled_features = (
                filling("feature_propagation", data.edge_index, x, missing_feature_mask, 40, )
            )
            # return the differences.
            frame1 = data.x.cpu().detach().numpy()  # original data.
            frame2 = np.nan_to_num(filled_features.cpu().detach().numpy())  # filled data.
            # x = dp.z_score(pd.DataFrame(x.cpu().detach().numpy()))
            frame3 = np.nan_to_num(x)  # unfilled data.
            # normalize all frames.
            frame1 /= np.linalg.norm(frame1)
            frame2 /= np.linalg.norm(frame2)
            frame3 /= np.linalg.norm(frame3)
            # calculating the difference.
            n1 = np.linalg.norm(frame1 - frame2)
            n2 = np.linalg.norm(frame1 - frame3)
            diff_filled[i - 1] += n1
            diff_unfilled[i - 1] += n2
            continue
        # the data is tabular.
        data_ = pd.read_csv(f"data/{name}.csv")
        data = data_.copy()
        # shuffle.
        data = shuffle(data)
        # normalize.
        data = dp.z_score(data)
        data, mask = dp.remove_random_cells(data, i / 10)
        x = data.drop(data.columns[-1], axis=1)
        # creating edges.
        if graph_style == "knn":
            edges, distances = mg.make_graph_knn(x, k=k)
            d = mg.find_mean_dist(x, edges)
        elif graph_style == "random":
            edges = mg.make_graph_random(x)
            d = mg.find_mean_dist(x, edges)
        elif graph_style == "forward":
            edges = mg.make_graph_forward(x)
            d = mg.find_mean_dist(x, edges)
        if print_dist_data:
            print(f"mean distance for method: {graph_style}, with missing rate {i / 5}% is: {np.mean(d):.3f}"
                  f"+-{np.std(d):.3f}")
        # filling.
        x = torch.from_numpy(x.values.astype(np.float32)).to(device)
        filled_features = (
            filling("feature_propagation", edges, x, mask, 40, )
        )
        # return the differences.
        frame1 = data_.iloc[:, :-1].values  # original data.
        frame2 = np.nan_to_num(filled_features.cpu().detach().numpy())  # filled data.
        frame3 = np.nan_to_num(x.cpu().detach().numpy())  # unfilled data.
        # normalize all frames.
        frame1 /= np.linalg.norm(frame1)
        frame2 /= np.linalg.norm(frame2)
        frame3 /= np.linalg.norm(frame3)
        # calculating the difference.
        n1 = np.linalg.norm(frame1 - frame2)
        n2 = np.linalg.norm(frame1 - frame3)
        diff_filled[i - 1] += n1
        diff_unfilled[i - 1] += n2

        # printing results.

        print("missing rate: ", i * 10, "%")
        print(f"the difference between the original data to the filled and unfilled data is: "
              f"{diff_filled[i - 1]:.3f} and {diff_unfilled[i - 1]:.3f} respectively.")

    plt.plot(range(10, 100, 10), diff_unfilled)
    plt.plot(range(10, 100, 10), diff_filled)
    plt.grid()
    plt.xlabel("missing rate")
    plt.ylabel("difference")
    plt.legend(["unfilled", "filled"])
    plt.show()

    return diff_unfilled, diff_filled


def test_filling_method(name: str, rates: list, k: int = 3, print_distance_info: bool = False):
    """
    this method loops over all missing rates and returns the results of xgv with and without the filling data.
    :param name: name of the dataset.
    :param rates: list of missing rates.
    :param k: used for knn.
    :param print_distance_info: flag deciding whether to print the mean distance data.
    :return: results of xgb.
    """
    a = np.zeros(len(rates))
    b = np.zeros(len(rates))
    for i, rate in enumerate(rates):
        print(f"working on {name} with missing rate {int(rate * 100)}%")
        d_u, d_f = remove_and_fill(name, print_distance_info, rate, k)

        # split both sets.
        filled_train, filled_test = train_test_split(d_f, test_size=0.2)
        unfilled_train, unfilled_test = train_test_split(d_u, test_size=0.2)

        # run.
        _, a_ = classify.run_xgb(unfilled_train, unfilled_test)
        _, b_ = classify.run_xgb(filled_train, filled_test)
        a[i] = a_
        b[i] = b_

    return a, b


def test_data(name: str):
    data_ = pd.read_csv(f"data/{name}.csv")
    data = data_.copy()
    for i in range(data.shape[1]):
        x = data.iloc[:, i]
        print(np.median(x), np.mean(x), np.std(x))
        plt.hist(x, 100)
        plt.ylabel("Frequency in column " + str(i))
        plt.show()
        plt.clf()


def test_distances(name: str, rate: float = 0.1, add_edge_holders: bool = False):
    """
    this method plots 2 scatter plots. in both plots the x-axis is the distances between the original data,
    and the y-axis is the distances between the missing data. in one plot the metric is Euclidean, and in the other
    it's the heuristic distance.
    :param name: name of the dataset.
    :return: 2 scatter plots describing the distances
    before and after removal. note: the x-axis is the same on both plots, given that heuristic distance and Euclidean
    distance are the same when no values are missing.
    """
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset, _ = data_loading.get_dataset(name)
        data = dataset.data
        n_nodes, n_features = data.x.shape

        missing_feature_mask = utils.get_missing_feature_mask(
            rate=rate, n_nodes=n_nodes, n_features=n_features, type="uniform",
        ).to(device)
        x = data.x.clone()
        x[~missing_feature_mask] = np.nan
        x_ = pd.DataFrame(data.x.clone().cpu().detach().numpy())

    # else: the dataset is tabular, hence we need to create edges.
    else:
        data_ = pd.read_csv(f"data/{name}.csv")

        # shuffle.
        data_ = shuffle(data_)
        x_ = data_.copy()
        x_ = dp.z_score(x_)
        x_ = x_.drop(x_.columns[-1], axis=1)
        # normalize.
        data = data_.copy()
        data = dp.z_score(data)
        data, mask = dp.remove_random_cells(data, rate)

        x = data.drop(data.columns[-1], axis=1)

        x = torch.from_numpy(x.values.astype(np.float32))

    x = pd.DataFrame(x.cpu().detach().numpy())

    # calculates distances for the original data.
    """
        for this part we calculate the knn distances for the missing data.
        then we'll consider the edges of knn to be the worthy distances. 
        therefore we have now a list of indices to calculate the distances between.
    """
    z = pd.DataFrame(x.copy().notna().astype(int))
    z = pd.concat([x, z], axis=1)

    # case 1: heuristic distance.
    edges_0, distances = mg.make_graph_knn(z, )

    # case 2: Euclidean distance.
    edges_1, distances_ = mg.make_graph_knn(z, dist="euc")

    # case 3: Euclidean distance without nans.
    edges_2, distances_2 = mg.make_graph_knn(z, dist="l_n")

    # case 4: custom metric.

    c = np.zeros(1)
    c[0] = 1.2207e-5
    distances_3 = calc_l2_normal(x, c)
    distances_3 = np.sqrt(distances_3)
    edges_3 = mg.get_knn_edges(distances_3, 3)
    # calculates the distances for the original data.
    d1 = calc_l2(x_)

    # calculating distances for edge holders.
    if add_edge_holders:
        d_edges_0 = [distances[edges_0[0, i], edges_0[1, i]] for i in range(edges_0.shape[1])]
        d_edges_1 = [distances_[edges_1[0, i], edges_1[1, i]] for i in range(edges_1.shape[1])]
        d_edges_2 = [distances_2[edges_2[0, i], edges_2[1, i]] for i in range(edges_2.shape[1])]
        d_edges_3 = [distances_3[edges_3[0, i], edges_3[1, i]] for i in range(edges_3.shape[1])]
        original_dists_0 = []
        original_dists_1 = []
        original_dists_2 = []
        original_dists_3 = []

        for i in range(edges_0.shape[1]):
            original_dists_0.append(d1[edges_0[0, i], edges_0[1, i]])
        for i in range(edges_1.shape[1]):
            original_dists_1.append(d1[edges_1[0, i], edges_1[1, i]])
        for i in range(edges_2.shape[1]):
            original_dists_2.append(d1[edges_2[0, i], edges_2[1, i]])
        for i in range(edges_3.shape[1]):
            original_dists_3.append(d1[edges_3[0, i], edges_3[1, i]])

    d1 = d1[np.triu_indices(d1.shape[0], k=1)].flatten()
    distances = distances[np.triu_indices(distances.shape[0], k=1)].flatten()
    distances_ = distances_[np.triu_indices(distances_.shape[0], k=1)].flatten()
    distances_2 = distances_2[np.triu_indices(distances_2.shape[0], k=1)].flatten()
    distances_3 = distances_3[np.triu_indices(distances_3.shape[0], k=1)].flatten()
    # NOTE: will only use 50% of the distances.
    d1 = d1[:int(len(d1) / 2)]
    distances = distances[:int(len(distances) / 2)]
    distances_ = distances_[:int(len(distances_) / 2)]
    distances_2 = distances_2[:int(len(distances_2) / 2)]
    distances_3 = distances_3[:int(len(distances_3) / 2)]

    # plot the scatter plots:
    s = [1] * len(d1)
    if add_edge_holders:
        s_0 = [1] * len(d_edges_0)
        s_1 = [1] * len(d_edges_1)
        s_2 = [1] * len(d_edges_2)
        s_3 = [1] * len(d_edges_3)

    plt.plot([min(d1), max(d1)], [min(d1), max(d1)], color="red", )
    plt.scatter(d1, distances, s=s)
    if add_edge_holders:
        plt.scatter(original_dists_0, d_edges_0, s=s_0, c="black")
    plt.title("Heuristic distance")
    plt.xlabel("Distance between original data")
    plt.ylabel("Distance between missing data")
    plt.grid()
    plt.savefig("plots/heuristic_distance.png")
    plt.show()
    plt.clf()

    plt.plot([min(d1), max(d1)], [min(d1), max(d1)], color="red")
    plt.scatter(d1, distances_, s=s)
    if add_edge_holders:
        plt.scatter(original_dists_1, d_edges_1, s=s_1, c="black")
    plt.title("Euclidean distance")
    plt.xlabel("Distance between original data")
    plt.ylabel("Distance between missing data")
    plt.grid()
    plt.savefig("plots/euclidean_distance.png")
    plt.show()
    plt.clf()

    plt.plot([min(d1), max(d1)], [min(d1), max(d1)], color="red")
    plt.scatter(d1, distances_2, s=s)
    if add_edge_holders:
        plt.scatter(original_dists_2, d_edges_2, s=s_2, c="black")
    plt.title("Euclidean distance without nans")
    plt.xlabel("Distance between original data")
    plt.ylabel("Distance between missing data")
    plt.grid()
    plt.savefig("plots/euclidean_distance_without_nans.png")
    plt.show()
    plt.clf()

    plt.plot([min(d1), max(d1)], [min(d1), max(d1)], color="red")
    plt.scatter(d1, distances_3, s=s)
    if add_edge_holders:
        plt.scatter(original_dists_3, d_edges_3, s=s_3, c="black")
    plt.title("Custom distance")
    plt.xlabel("Distance between original data")
    plt.ylabel("Distance between missing data")
    plt.grid()
    plt.savefig("plots/custom_distance.png")
    plt.show()


def compare_methods(name: str, rate: float = 0.1, ):
    """
    this method plots a scatter plot of the difference between the custom metric and the filled distances,
    one scatter plot according to the true edges, and the other according to the metric's edges.
    :param name: name of the dataset.
    :param rate: missing values rate.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    # z-score.
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)

    # remove data.
    data, mask = dp.remove_random_cells(data_, rate)
    data = data.drop(data.columns[-1], axis=1)

    c = np.zeros(1)
    c[0] = 1.2207e-5
    missing_distances = calc_l2_normal(data, c)
    missing_distances = np.sqrt(missing_distances)

    edges = mg.get_knn_edges(distances_full, 3)
    # get only edge holders.
    full_dists = find_top_candidates(distances_full, edges)
    missing_dists = find_top_candidates(missing_distances, edges)

    # plot scatter plot.
    min_ = min(min(full_dists), min(missing_dists))
    max_ = max(max(full_dists), max(missing_dists))
    plt.plot([min_, max_], [min_, max_], color="red")
    plt.scatter(full_dists, missing_dists)
    plt.title("Differences with true edges")
    plt.xlabel("Distance between original data")
    plt.ylabel("Distance between missing data")
    plt.grid()
    plt.savefig("plots/true_edges_diff.png")
    plt.show()
    plt.clf()

    # now change the edges.
    edges = mg.get_knn_edges(missing_distances, 3)
    # get only edge holders.
    missing_dists = find_top_candidates(missing_distances, edges)
    full_dists = find_top_candidates(distances_full, edges)

    # plot scatter plot.
    min_ = min(min(full_dists), min(missing_dists))
    max_ = max(max(full_dists), max(missing_dists))
    plt.plot([min_, max_], [min_, max_], color="red")
    plt.scatter(full_dists, missing_dists)
    plt.title("Differences with metric's edges")
    plt.xlabel("Distance between original data")
    plt.ylabel("Distance between missing data")
    plt.grid()
    plt.savefig("plots/metric_edges_diff.png")
    plt.show()


def get_stats(name: str):
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    rates = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25]

    # for each rate, calculate distances before & after the removal.
    k_0 = []
    k_1 = []
    k_2 = []
    k_0_after = []
    k_1_after = []
    k_2_after = []

    p = lambda k: (1 - k) ** x_.shape[1]  # sample is unchanged with change rate k.
    f_0 = lambda k: (p(k)) ** 2  # a pair is unchanged.
    f_1 = lambda k: 2 * p(k) * (1 - p(k))  # only one is changed within a pair.
    f_2 = lambda k: (1 - p(k)) ** 2  # both are changed within a pair.

    for rate in rates:
        data = data_.copy()
        data = dp.z_score(data)

        data, mask = dp.remove_random_cells(data, rate)
        # working only with euc dist.
        x = data.drop(data.columns[-1], axis=1)

        k_0.append(f_0(rate))
        k_1.append(f_1(rate))
        k_2.append(f_2(rate))

        vals = []
        z = pd.DataFrame(x.copy().notna().astype(int))
        for i in range(x.shape[0]):
            # check if the row contains a nan value.
            vals.append(sum(z.iloc[i]) == x.shape[1])
        # find % of row pairs that stayed the same.
        k_0_counter = 0
        k_1_counter = 0
        k_2_counter = 0
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                if i == j:
                    continue
                if vals[i] and vals[j]:
                    k_0_counter += 1
                elif vals[i] or vals[j]:
                    k_1_counter += 1
                else:
                    k_2_counter += 1

        n = x.shape[0] - 1

        k_0_after.append(k_0_counter * 2 / ((n - 1) ** 2))
        k_1_after.append(k_1_counter * 2 / ((n - 1) ** 2))
        k_2_after.append(k_2_counter * 2 / ((n - 1) ** 2))

    plt.scatter(k_0, k_0_after, marker="o")
    m = max(max(k_0), max(k_0_after))
    plt.plot([0, m], [0, m], color="red")
    plt.grid()
    plt.xlabel("theoretical % of unchanged pairs")
    plt.ylabel("actual % of unchanged pairs")
    plt.title("0 changes")
    plt.savefig("plots/0.png")
    plt.show()
    plt.clf()
    print(k_0)
    print(k_0_after)

    plt.scatter(k_1, k_1_after, marker="x")
    m = max(max(k_1), max(k_1_after))
    plt.plot([0, m], [0, m], color="red")
    plt.grid()
    plt.xlabel("theoretical % of unchanged pairs")
    plt.ylabel("actual % of unchanged pairs")
    plt.title("1 change")
    plt.savefig("plots/1.png")
    plt.show()
    plt.clf()
    print(k_1)
    print(k_1_after)

    plt.scatter(k_2, k_2_after, marker="^")
    m = max(max(k_2), max(k_2_after))
    plt.plot([0, m], [0, m], color="red")
    plt.grid()
    plt.xlabel("theoretical % of unchanged pairs")
    plt.ylabel("actual % of unchanged pairs")
    plt.title("2 changes")
    plt.savefig("plots/2.png")
    plt.show()
    plt.clf()
    print(k_2)
    print(k_2_after)


def test_data_removal(name: str):
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    rates = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
    k = []
    k_after = []
    p = lambda j: (1 - j) ** x_.shape[1]  # sample is unchanged with change rate k.
    f_0 = lambda j: (p(j))  # a pair is unchanged.

    for rate in rates:
        data = data_.copy()
        data = dp.z_score(data)
        data, mask = dp.remove_random_cells(data, rate)
        # working only with euc dist.
        x = data.drop(data.columns[-1], axis=1)

        k.append(f_0(rate))

        vals = []
        z = pd.DataFrame(x.copy().notna().astype(int))
        for i in range(x.shape[0]):
            # check if the row contains a nan value.
            vals.append(1 if sum(z.iloc[i]) == x.shape[1] else 0)
        # find % of row pairs that stayed the same.

        k_after.append(sum(vals) / len(vals))

    plt.scatter(k, k_after, marker="o")
    m = max(max(k), max(k_after))
    plt.plot([0, m], [0, m], color="red")
    plt.grid()
    plt.xlabel("theoretical % of unchanged pairs")
    plt.ylabel("actual % of unchanged pairs")
    plt.title("0 changes")
    plt.savefig("plots/0_rows.png")
    plt.show()
    print(k)
    print(k_after)


def get_custom_distances(features: pd.DataFrame):
    """
    this method calculates the pairwise distances between each pair of rows in the missing data.
    :param features: the data.
    :return: matrix of distances, sized n x n when n is the number of rows.
    the calculation is done using estimated Euclidean distance.
    It's assumed that the data has been z-scored (using the z_score method above, or any other z-scoring technique).
    d(x,y) = sqrt(sum((x_i - y_i)^2) + |x_f or y_f| + |x_f and y_f|)
    """
    mask_missing = features.isna().astype(int)  # get mask where 1 means nan.
    dists = np.square(calc_l2(features))  # get squared L2 distances.
    dists += np.square(calc_l2(mask_missing))  # add the size of the xor between each pair. (only one missing)
    dists += 2 * np.matmul(mask_missing, mask_missing.T)  # add the size of the intersection between each pair.
    # (both missing)
    return np.sqrt(dists)


def calc_l2(data: pd.DataFrame):
    """
    This method calculates the L2 distances between rows in the data when NaN values are replaced with 0.
    :param data: the data.
    :return: matrix of distances.
    """
    data = data.fillna(0)
    dists = np.linalg.norm(data.values[:, np.newaxis, :] - data.values[np.newaxis, :, :], axis=-1)

    return dists


def calc_l2_neglecting_nans(data: pd.DataFrame):
    """
    This method calculates the L2 distances between rows in the data when NaN values are neglected.
    :param data: the data.
    :return: matrix of distances.
    """
    data = data.fillna(np.nan)  # Fill NaN values to ensure consistent handling
    data = data.values
    num_rows, num_cols = data.shape
    distances = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            mask_numeric = ~np.isnan(data[i]) & ~np.isnan(data[j])

            if np.sum(mask_numeric) == 0:
                continue

            distances[i, j] = np.linalg.norm(data[i, mask_numeric] - data[j, mask_numeric])
            distances[i, j] *= distances[i, j] * data.shape[1] / np.sum(mask_numeric)
            distances[i, j] = np.sqrt(distances[i, j])
    return distances


def calc_heur_dists(data: pd.DataFrame):
    """
    this method calculates the heuristic distances between rows in the data.
    :param data: the data.
    :return: list of distances.
    """
    z = pd.DataFrame(data.copy().notna().astype(int))
    z = pd.concat([data, z], axis=1)
    _, dists = mg.make_graph_knn(z)

    return dists


def plot_dists_hist(name: str, rate: float = 0.1, use_line_plot: bool = False):
    """
    this method calculates and plots histogram of the differences between the distances of the original data and the
    novel metrics above.
    :param use_line_plot: whether to use line plot or histogram.
    :param rate: missing values rate.
    :param name: name of the dataset.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    distances_full = calc_l2(x_)

    # remove data.
    data, mask = dp.remove_random_cells(data_, rate)
    data = data.drop(data.columns[-1], axis=1)  # keep only features.
    # calculate distances.
    d1 = calc_l2(data)
    d2 = calc_l2_neglecting_nans(data)
    d3 = calc_heur_dists(data)

    # calculate lists of the squared differences between the original distances and each of the metrics.
    d1 = np.square(d1) - np.square(distances_full)
    d2 = np.square(d2) - np.square(distances_full)
    d3 = np.square(d3) - np.square(distances_full)

    # keep only the part of the matrix that is above the diagonal. not including the diagonal itself.
    d1 = d1[np.triu_indices(d1.shape[0], k=1)].flatten()
    d2 = d2[np.triu_indices(d2.shape[0], k=1)].flatten()
    d3 = d3[np.triu_indices(d3.shape[0], k=1)].flatten()

    # find mean and std for each histogram.
    m1 = np.mean(d1)
    m2 = np.mean(d2)
    m3 = np.mean(d3)
    std1 = np.std(d1)
    std2 = np.std(d2)
    std3 = np.std(d3)

    # plot histograms.
    d1_nums, _, d1_patches = plt.hist(d1, 50, range=(-10, 10))
    plt.title("L2 distance, mean: {:.3f}, std: {:.3f}".format(m1, std1))
    plt.xlabel("Squared difference")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("plots/l2.png")
    if not use_line_plot:
        plt.show()
    plt.clf()

    d2_nums, _, d2_patches = plt.hist(d2, 50, range=(-10, 10))
    plt.title("L2 distance neglecting nans, mean: {:.3f}, std: {:.3f}".format(m2, std2))
    plt.xlabel("Squared difference")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("plots/l2_neglecting_nans.png")
    if not use_line_plot:
        plt.show()
    plt.clf()

    d3_nums, bins, d3_patches = plt.hist(d3, 50, range=(-10, 10))
    plt.title("Heuristic distance, mean: {:.3f}, std: {:.3f}".format(m3, std3))
    plt.xlabel("Squared difference")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("plots/heuristic.png")
    if not use_line_plot:
        plt.show()
    plt.clf()
    # print each mean and std.
    print(f"mean of l2: {np.mean(d1):.3f} +- {np.std(d1):.3f}")
    print(f"mean of l2 neglecting nans: {np.mean(d2):.3f} +- {np.std(d2):.3f}")
    print(f"mean of heuristic: {np.mean(d3):.3f} +- {np.std(d3):.3f}")

    if use_line_plot:
        plt.plot(bins[:-1], d1_nums, "b-", label="L2")
        plt.plot(bins[:-1], d2_nums, "r-", label="L2 neglecting nans")
        plt.plot(bins[:-1], d3_nums, "g-", label="Heuristic")
        plt.legend()
        plt.grid()
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("plots/line_plot.png")
        plt.show()


def get_best_constants(name: str, rates: list, iters: int = 10):
    """
    this method calculates the correlation between the top 10% the smallest distances in the original data and the
    distances between the chosen rows in the new metric, and uses nedler-mead optimizer to find the fittest c_1,
    c_2 to maximize the intersection.
    :param iters: num of iterations.
    :param name: name of the dataset.
    :param rates: rates of missing data.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    # z-score.
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)

    # use knn method to find the nodes between which the distances are smallest.
    edges_true = mg.get_knn_edges(distances=distances_full, k=40)
    a = edges_true.t().numpy()

    params_list = np.zeros((len(rates), 2))
    for idx, rate in enumerate(rates):
        temp_params = np.zeros((iters, 2))  # save results from all executions in order to find mean and std.
        for i in range(10):
            print("rate ", rate, " iteration ", (i + 1))
            # remove data.
            data, mask = dp.remove_random_cells(data_.copy(), rate)
            data = data.drop(data.columns[-1], axis=1)

            x = data.drop(data.columns[-1], axis=1)
            # collect edges for unfilled data graph.
            edges = []
            for j in range(x.shape[1]):
                dists = metric_by_feature(x, j)
                edges.extend(mg.get_knn_edges(dists, 10).t().numpy().tolist())

            edges = np.unique(np.array(edges), axis=0)

            # fill with regression.
            data = test_regression_features_(data)
            dists = calc_l2(data)

            mask_vals = ~mask.cpu().numpy().astype(int)

            d_params = lambda c: (
                    -len(find_intersection(a, np.array(list(
                        find_intersection(edges, mg.get_knn_edges(
                            fixed_metric(dists, mask_vals, c), 40)))))))

            # Initial guess for c
            initial_params = np.array([1, 1])

            # optimize c1, c2, using nedler-mead.
            res = minimize(d_params, initial_params, method="Nelder-Mead", options={"disp": True})

            temp_params[i] = res.x

        # fix params.
        for i in range(2):
            params_list[idx, i] = np.mean(temp_params[:, i])

        # remove data.
        data, mask = dp.remove_random_cells(data_.copy(), rate)

        x = data.drop(data.columns[-1], axis=1)
        # collect edges for unfilled data graph.
        edges = []
        for j in range(x.shape[1]):
            dists = metric_by_feature(x, j)
            edges.extend(mg.get_knn_edges(dists, 10).t().numpy().tolist())

        edges = np.unique(np.array(edges), axis=0)

        # fill with regression.
        data = test_regression_features_(data)

        dists = calc_l2(data)
        dists = fixed_metric(dists, mask.cpu().numpy(), params_list[idx])
        edges_ = mg.get_knn_edges(dists, 40).t().numpy()
        edges_ = np.unique(edges_, axis=0)
        edges = np.array(list(find_intersection(edges, edges_)))

        print("Best params for rate " + str(int(100 * rate)) + "%:")
        for i in range(2):
            print(f"c{i}: {params_list[idx][i]:.3f} +- {np.std(temp_params[:, i]):.3f}")
        print("\nUnion: ", len(find_union(a, edges)))
        print("Intersection: ", len(find_intersection(a, edges)))
        print("Symmetric difference: ", len(find_uniques(a, edges)))


def find_top_candidates(distances, edges):
    """
    this method finds the distances of the knn graph.
    :param edges:
    :param distances:
    :return: list of distances between the edge holders.
    """
    return [distances[edges[0, i], edges[1, i]] for i in range(edges.shape[1])]


def calc_l2_normal(size: int, t_t, xor_vals, t_f, f_t, truth_mask, c: np.ndarray):
    """
    This method calculates the L2 distances between rows in the data when NaN values are neglected.
    :param truth_mask: mask of intersection sizes between all row pairs.
    :param size: num of features.
    :param f_t: how many features exist in the 2nd and not 1st in each pair i,j.
    :param t_f: how many features exist in the 1st and not 2nd in each pair i,j.
    :param xor_vals: sum of squares of values present in only one of the rows in each pair i,j.
    :param t_t: l2 distances where all values are present.
    :param c: penalty coefficients.
    :return: matrix of distances.
    """
    t_t[truth_mask == 0] = max(t_t.flatten())  # replace full nan rows with max value.
    t_t[truth_mask != 0] = t_t[truth_mask != 0] * np.power(size / truth_mask[truth_mask != 0], c[0])

    return t_t + c[1] * xor_vals + c[2] * t_f + c[3] * f_t


def find_uniques(a: np.ndarray, b: np.ndarray):
    """
    this method takes in two matrices and returns a set of the row-wise symmetric difference between a and b.
    :param a: 1st matrix.
    :param b: 2nd matrix.
    :return: set of unique rows.
    """
    a = set(tuple(x) for x in a)
    b = set(tuple(x) for x in b)
    return a.symmetric_difference(b)


def find_union(a: np.ndarray, b: np.ndarray):
    """
    this method takes in two matrices and returns a set of the row-wise union between a and b.
    :param a: 1st matrix.
    :param b: 2nd matrix.
    :return: set of unique rows.
    """
    a = set(tuple(x) for x in a)
    b = set(tuple(x) for x in b)
    return a.union(b)


def find_intersection(a: np.ndarray, b: np.ndarray):
    """
    this method takes in two matrices and returns a set of the row-wise intersection between a and b.
    :param a: 1st matrix.
    :param b: 2nd matrix.
    :return: set of unique rows.
    """
    a = set(tuple(x) for x in a)
    b = set(tuple(x) for x in b)
    return a.intersection(b)


def find_diff(a: np.ndarray, b: np.ndarray):
    """
    this method takes in two matrices and returns a set of the row-wise difference between a and b.
    :param a: 1st matrix.
    :param b: 2nd matrix.
    :return: set of unique rows.
    """
    a = set(tuple(x) for x in a)
    b = set(tuple(x) for x in b)
    return a.difference(b)


def test_many_methods(name: str):
    """
    this method tests the methods of filling and removing data.
    :param name: name of the dataset.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    # z-score.
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)

    # remove data.
    data, mask = dp.remove_random_cells(data_, 0.1)
    data = data.drop(data.columns[-1], axis=1)

    # use knn method to find the nodes between which the distances are smallest.
    edges_true = mg.get_knn_edges(distances=distances_full, k=3)
    a = edges_true.t().numpy()

    # calculate the amount of edges that are present in both the original data and the new metric.
    diff = lambda c: len(find_uniques(a, mg.get_knn_edges(mg.custom_distance_matrix(data, c[0], c[1]), 3).t().numpy()))

    # calculates pairwise distances according to the metrics.
    n = data.shape[0]
    dists = {
        1: np.zeros((n, n)),  # using xor
        2: np.zeros((n, n)),  # using sum of squares
        3: np.zeros((n, n)),  # using false values as penalty
        4: np.zeros((n, n)),  # using multiplication of false indexes
        5: np.zeros((n, n)),  # using division
        6: np.zeros((n, n)),  # using exponential
        7: np.zeros((n, n)),  # using addition of false indexes
        8: np.zeros((n, n)),  # using multiplication of false indexes
        9: np.zeros((n, n)),  # using custom distance
        10: np.zeros((n, n)),  # using normalized l2.
    }
    data_vals = data.values
    for i in range(n):
        for j in range(i, n):
            i_nums = ~np.isnan(data_vals[i])
            j_nums = ~np.isnan(data_vals[j])
            xor = np.logical_xor(i_nums, j_nums)
            intersection = np.logical_and(i_nums, j_nums)
            true_diff = np.square(np.linalg.norm(data_vals[i, intersection] - data_vals[j, intersection]))

            sum_i_nans = np.sum(~i_nums)
            sum_j_nans = np.sum(~j_nums)
            intersection_nans = np.logical_and(~i_nums, ~j_nums)

            # add penalties depending on the metric.
            dists[1][i, j] = np.sum(xor) + true_diff
            dists[2][i, j] = np.square(sum_i_nans) + np.square(sum_j_nans) + true_diff
            dists[3][i, j] = (np.sum(np.square(data_vals[i, ~j_nums])) + np.sum(np.square(data_vals[j, ~i_nums])) +
                              true_diff)
            dists[4][i, j] = (sum_i_nans + 1) * (sum_j_nans + 1) - 1 + true_diff
            dists[5][i, j] = ((sum_i_nans * sum_j_nans + sum_i_nans + sum_j_nans) / (sum_i_nans + sum_j_nans + 1) +
                              true_diff)
            dists[6][i, j] = np.exp(np.sum(intersection_nans)) - 1 + true_diff
            dists[7][i, j] = sum_i_nans + sum_j_nans + true_diff
            dists[8][i, j] = (sum_i_nans * sum_j_nans + sum_i_nans + sum_j_nans) + true_diff
            dists[10][i, j] = true_diff * data.shape[1] / np.sum(intersection) + (data.shape[1] - np.sum(intersection)) \
                if np.sum(intersection) > 0 else data.shape[1]

            for k in dists.keys():
                dists[k][j, i] = dists[k][i, j]

    # Initial guess for c
    initial_params = np.array([0, 0])

    # optimize c1, c2 using nedler-mead.
    res = minimize(diff, initial_params, method="Nelder-Mead", options={'disp': True})

    params = res.x
    dists[9] = mg.custom_distance_matrix(data, params[0], params[1])

    # score each function.
    dct = {}
    print(f"# of true edges: {a.shape[0]}")
    for k in dists.keys():
        edges = mg.get_knn_edges(dists[k], 3).t().numpy()
        print(f"# of edges for function {k}: {edges.shape[0]}")
        inter = len(find_intersection(a, edges))
        xor = len(find_uniques(a, edges))
        union = len(find_union(a, edges))
        print(f"intersection for function {k}: {inter}")
        print(f"xor for function {k}: {xor}")
        print(f"union for function {k}: {union}")
        # save the tuple of (intersection, xor, union) for each function.
        dct[k] = (inter, xor, union)

    # make dataframe from results and save to csv.
    df = pd.DataFrame.from_dict(dct, orient="index", columns=["intersection", "xor", "union"])
    df.to_csv("results.csv")


def calc_masks(data: pd.DataFrame):
    """
    this method takes a dataframe with missing values, and calculates the following distances:
    l2 when both values are missing, sum of squares when only one is missing, mask when only one is missing.
    :param data: the dataset.
    :return: (t_t, xor_vals, t_f, f_t) masks.
    """
    data = data.values
    mask = ~np.isnan(data)
    data = np.nan_to_num(data)  # fill nan with 0.
    n = data.shape[0]
    t_t = np.zeros((n, n))
    xor_vals = np.zeros((n, n))
    t_f = np.zeros((n, n))
    f_t = np.zeros((n, n))
    truth_mask = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            intersection = mask[i] & mask[j]
            only_i = mask[i] & ~mask[j]
            only_j = mask[j] & ~mask[i]
            t_t[i, j] = np.sum(np.square(data[i, intersection] - data[j, intersection]))  # l2 where both are present.
            xor_vals[i, j] = np.sum(np.square(data[i, only_i])) + np.sum(np.square(data[j, only_j]))  # sum of
            # squares where only one is.
            t_f[i, j] = np.sum(only_i)
            f_t[i, j] = np.sum(only_j)
            truth_mask[i, j] = np.sum(intersection)

            # make all symmetric.
            t_t[j, i] = t_t[i, j]
            xor_vals[j, i] = xor_vals[i, j]
            t_f[j, i] = t_f[i, j]
            f_t[j, i] = f_t[i, j]
            truth_mask[j, i] = truth_mask[i, j]

    return t_t, xor_vals, t_f, f_t, truth_mask


def keep_only_uniques(edges: torch.tensor):
    """
    this method takes in a tensor of edges, and returns a tensor of the same edges, but with only the unique edges.
    :param edges: tensor of edges.
    :return: tensor of unique edges.
    """
    edges = edges.t().numpy()
    edges = set(tuple(x) for x in edges)
    return torch.tensor(np.array(list(edges))).t()


def plot_scatter_with_divisions(name: str, rates: list, k: int = 3):
    """
    this method plots a scatter plot of the differences between the custom metric on the missing data, and l2 on the
    full data, and colors the points according to the intersection and set differences between the knn edges.
    :param k: for knn.
    :param name: name of the dataset.
    :param rates: list of rates.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    # z-score.
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)
    # use knn method to find the nodes between which the distances are smallest.
    edges_true = mg.get_knn_edges(distances=distances_full, k=k)
    a = edges_true.t().numpy()

    # for each rate, calculate distances.
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, rate in enumerate(rates):
        # remove data.
        print(f"rate: {rate}")
        data, mask = dp.remove_random_cells(data_.copy(), rate)

        data = test_regression_features_(data)
        data = data.drop(data.columns[-1], axis=1)

        dists = calc_l2(data)

        # calculate edges.
        edges = mg.get_knn_edges(dists, k)
        b = edges.t().numpy()

        # find the intersection and set differences between the edges.
        inter = np.array(list(find_intersection(a, b)))
        only_true = np.array(list(find_union(a, b).difference(set(tuple(x) for x in b))))  # edges that are
        # present only in the true edges.
        only_metric = np.array(list(find_union(a, b).difference(set(tuple(x) for x in a))))  # edges that are
        # present only in the metric edges.

        # save the distances according to the presence of the edges.
        dists_true_x = [distances_full[x[0], x[1]] for x in only_true]
        dists_true_y = [dists[x[0], x[1]] for x in only_true]
        dists_metric_x = [distances_full[x[0], x[1]] for x in only_metric]
        dists_metric_y = [dists[x[0], x[1]] for x in only_metric]
        dists_inter_x = [distances_full[x[0], x[1]] for x in inter]
        dists_inter_y = [dists[x[0], x[1]] for x in inter]

        # plot the scatters.
        axs[i // 3, i % 3].scatter(dists_true_x, dists_true_y, c="red", label="True edges", marker="^")
        axs[i // 3, i % 3].scatter(dists_metric_x, dists_metric_y, c="green", label="Metric-based edges", marker="o")
        axs[i // 3, i % 3].scatter(dists_inter_x, dists_inter_y, c="black", label="Intersection", marker="x")
        axs[i // 3, i % 3].legend(loc="upper right")
        axs[i // 3, i % 3].set_title(f"Rate: {rate}, K={k}, Intersection: {len(inter)}")
        axs[i // 3, i % 3].set_xlabel("Original difference")
        axs[i // 3, i % 3].set_ylabel("Removed difference")
        axs[i // 3, i % 3].grid()

    plt.savefig(f"plots/scatter_{name}_.png")
    fig.show()


# implementing union find for connection components.
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def find_components(num_nodes, edges):
    uf = UnionFind(num_nodes)

    for edge in edges.t():
        uf.union(edge[0].item(), edge[1].item())

    component_sizes = {}
    for i in range(num_nodes):
        root = uf.find(i)
        if root in component_sizes:
            component_sizes[root] += 1
        else:
            component_sizes[root] = 1

    roots = np.array([k for k in component_sizes.keys()])
    sizes = np.array([v for v in component_sizes.values()])
    return roots, sizes


# plotting knn components histogram for each k and rate.
def plot_comp_hists(name: str, rates: list, k_vals: list):
    """
    this method plots a histogram of the sizes of the connected components in the knn graph, for each rate and for each
    k values.
    :param name: name of the dataset.
    :param rates: rates of missing values.
    :param k_vals: value of k for knn.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    # z-score.
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)

    # set up dictionary of metric constants.
    constants = {
        0.1: [0.839, 0, 1.366, 1.109],
        0.2: [0.882, 0, 1.2, 1.265],
        0.3: [0.963, 0, 1.074, 1.14],
        0.4: [1.027, 0, 1.035, 1.035],
        0.5: [1.026, 0, 1.026, 1.032],
        0.6: [1.026, 0, 1.026, 1.026],
        0.7: [1.026, 0, 0.996, 1.024],
        0.8: [1.024, 0, 0.958, 1.024],
        0.9: [0.981, 0, 1.014, 1.036],
    }

    for k in k_vals:
        print(f"K value: {k}")
        # use knn method to find the nodes between which the distances are smallest.
        edges_true = mg.get_knn_edges(distances=distances_full, k=k)
        r_t, s_t = find_components(data_.shape[0], edges_true)
        # plot histogram of true components.
        plt.bar(range(len(r_t)), s_t, color="red")
        plt.title(f"Histogram of true edges for K={k}")
        plt.xlabel("Component number")
        plt.ylabel("Component size")
        plt.grid()
        plt.savefig(f"plots/hist_true_comp-sizes_k_{k}.png")
        plt.show()

        # for each rate, calculate distances.
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for i, rate in enumerate(rates):
            # remove data.
            print(f"rate: {rate}")
            data, mask = dp.remove_random_cells(data_.copy(), rate)
            data = data.drop(data.columns[-1], axis=1)

            # calculate masks.
            t_t, xor_vals, t_f, f_t, truth_mask = calc_masks(data)

            # use the constants to calculate the distances.
            dists = calc_l2_normal(data.shape[1], t_t, xor_vals, t_f, f_t, truth_mask, np.array(constants[rate]))
            dists = np.sqrt(dists)

            # calculate edges.
            edges = mg.get_knn_edges(dists, k)

            # find the connection component sizes in both the original and removed graphs.
            r_m, s_m = find_components(data.shape[0], edges)

            # plot histograms.
            axs[i // 3, i % 3].bar(range(len(r_m)), s_m, color="green")
            axs[i // 3, i % 3].set_title(f"Rate: {rate}, k={k}, Largest component in %: "
                                         f"{max(s_m) / data.shape[0] * 100:.2f}")
            axs[i // 3, i % 3].set_xlabel("Component number")
            axs[i // 3, i % 3].set_ylabel("Component size")
            axs[i // 3, i % 3].grid()

        plt.savefig(f"plots/hist_data_comp-sizes_k_{k}.png")
        fig.show()


def calc_masks_with_false(data: pd.DataFrame):
    """
    this method is similar to calc_masks, but it also calculates the amount of nan pairs within each pair.
    :param data: the dataset features
    :return:
    """
    data = data.values
    mask = ~np.isnan(data)  # 1 for numbers, 0 for nan.
    data = np.nan_to_num(data)  # fill nan with 0.
    n = data.shape[0]
    t_t = np.zeros((n, n))
    f_f = np.zeros((n, n))
    t_f = np.zeros((n, n))
    f_t = np.zeros((n, n))
    truth_mask = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            intersection = mask[i] & mask[j]
            only_i = mask[i] & ~mask[j]
            only_j = mask[j] & ~mask[i]
            t_t[i, j] = np.sum(np.square(data[i, intersection] - data[j, intersection]))  # l2 where both are present.
            f_f[i, j] = np.sum(~mask[i] & ~mask[j])  # sum of
            # squares where only one is.
            t_f[i, j] = np.sum(only_i)
            f_t[i, j] = np.sum(only_j)
            truth_mask[i, j] = np.sum(intersection)

            # make all symmetric.
            t_t[j, i] = t_t[i, j]
            f_f[j, i] = f_f[i, j]
            t_f[j, i] = t_f[i, j]
            f_t[j, i] = f_t[i, j]
            truth_mask[j, i] = truth_mask[i, j]

    return t_t, np.square(f_f), t_f, f_t, truth_mask


def calc_new_metric(t_t, f_f, t_f, f_t, truth_mask, size, c):
    """
    this method calculates the new metric according to the formula:
    t_t * (size / truth_mask)**c[0] + f_f**2 * c[1] + t_f * c[2] + f_t * c[3]
    :param c: penalty coefficients
    :param t_t: true distances.
    :param f_f: mask of double nan pairs.
    :param t_f: mask where only the 2nd is nan.
    :param f_t: mask where only the 1st is nan.
    :param truth_mask: mask where there are number pairs.
    :param size: number of features.
    :return:
    """
    t_t[truth_mask == 0] = max(t_t.flatten())  # replace full nan rows with max value.
    t_t[truth_mask != 0] = t_t[truth_mask != 0] * np.power(size / truth_mask[truth_mask != 0], c[0])
    dists = t_t + c[1] * f_f + c[2] * t_f + c[3] * f_t
    return np.sqrt(dists)


def test_knn_from_mix(name: str, mix_size: list, rate: float = 0.9):
    """
    this method tests the xgb with FP, but the graph is built from mix_size neighbors from the real knn and 5 -
    mix_size from the metric knn.
    :param name: the dataset name.
    :param mix_size: the amount of neighbors from the
    real knn.
    :param rate: missing value rate.
    :return:
    """
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    # set up dictionary of metric constants.
    constants = {
        0.1: [0.839, 0, 1.366, 1.109],
        0.2: [0.882, 0, 1.2, 1.265],
        0.3: [0.963, 0, 1.074, 1.14],
        0.4: [1.027, 0, 1.035, 1.035],
        0.5: [1.026, 0, 1.026, 1.032],
        0.6: [1.026, 0, 1.026, 1.026],
        0.7: [1.026, 0, 0.996, 1.024],
        0.8: [1.024, 0, 0.958, 1.024],
        0.9: [0.981, 0, 1.014, 1.036],
    }

    results = {
        "no removal": 0,
        "5": 0,
        "4": 0,
        "3": 0,
        "2": 0,
        "1": 0,
        "0": 0,
        "unfilled": 0,
    }

    # remove data.
    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    y = data_.iloc[:, -1].copy()
    x_ = data_.copy().drop(data_.columns[-1], axis=1)

    data_true = data_.copy()
    true_train, true_test = train_test_split(data_true, test_size=0.2)
    results["no removal"] = classify.run_xgb(true_train, true_test)[1]  # run xgb on full data.

    data, mask = dp.remove_random_cells(data_, rate)

    x = data.drop(data.columns[-1], axis=1)
    unfilled_data = pd.concat([x, y], axis=1)
    unfilled_train, unfilled_test = train_test_split(unfilled_data, test_size=0.2)
    results["unfilled"] = classify.run_xgb(unfilled_train, unfilled_test)[1]  # run xgb on unfilled data.

    # calculate real edges.
    distances_full = calc_l2(x_)

    # calculate metric edges.
    t_t, xor_vals, t_f, f_t, truth_mask = calc_masks(x)
    dists = calc_l2_normal(data.shape[1], t_t, xor_vals, t_f, f_t, truth_mask, np.array(constants[rate]))
    dists = np.sqrt(dists)

    y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64))

    # mix the edges.
    for size in mix_size:
        # use knn on real distances with 'size', and knn on metric distances with 5 - size.
        neighbors = mg.get_knn_edges(distances_full, size).t().numpy()
        neighbors_metric = mg.get_knn_edges(dists, 5 - size).t().numpy()
        # add them to the use_edges list.
        use_edges = np.append(neighbors_metric, neighbors, axis=0)
        use_edges = torch.tensor(use_edges).t()
        # fill features.
        x = torch.from_numpy(x.copy().values.astype(np.float32)).to(device)

        filled_features = (
            filling("feature_propagation", use_edges, x, mask, 40, )
        )

        x = pd.DataFrame(x.cpu().numpy())

        filled_data = pd.DataFrame(filled_features.cpu().numpy())
        filled = pd.concat([filled_data, pd.DataFrame(y)], axis=1)

        # run xgb.
        train, test = train_test_split(filled, test_size=0.2)
        results[f"{size}"] = classify.run_xgb(train, test)[1]

    # print results.
    for (size, result) in results.items():
        print(f"{size} real edges: {result:.3f}")


def find_cov_matrix(data: pd.DataFrame):
    """
    this method takes a dataset with missing values, and calculates the covariance between each feature and all
    others, based on samples when both features exist.
    :param data:
    :return: np.ndarray of the covariance matrix.
    (sized m x m when m is the number of features)
    note-the dataset is assumed to be z-scored, and only containing the features.
    """
    data = data.values
    cov = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            ij_pairs = []
            for k in range(data.shape[0]):  # go over all rows.
                if not np.isnan(data[k, i]) and not np.isnan(data[k, j]):  # if both features exist in the same row.
                    ij_pairs.append([data[k, i], data[k, j]])  # add the pair of features to the list.
            # calculate the covariance between the two features.
            if len(ij_pairs) > 0:
                cov[i, j] = cov[j, i] = np.cov(np.array(ij_pairs).T)[0, 1]
            else:
                cov[i, j] = cov[j, i] = 0

    return cov


def run_pca(data: pd.DataFrame, cov_mat: np.array = None, ):
    """
    this method gets the covariance matrix of a dataset, and runs pca on it.
    :param data: the missing dataset.
    :param cov_mat: the covariance matrix of the dataset's features.
    :return:
    """
    if cov_mat is None:
        cov_mat = find_cov_matrix(data)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    # Step 4: Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select principal components
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    num_components_needed = np.argmax(cumulative_variance_ratio >= 0.9) + 1

    # Step 6: Project data onto subspace
    selected_eigenvectors = eigenvectors[:, :num_components_needed]
    X_reduced = np.dot(np.nan_to_num(data.values), selected_eigenvectors)
    X_reconstructed = np.dot(X_reduced, selected_eigenvectors.T)

    return X_reconstructed


def test_pca(name: str, rates: list):
    """
    this method uses the pca method as a middle step in imputing the data, and then runs xgb on the unfilled and imputed
    data.
    :param name: name of the dataset.
    :param rates: rates of missing values
    :return:
    """
    a, b = [], []
    for rate in rates:
        print(f"rate: {int(100 * rate)}")
        a_, b_ = test_pca_one_value(name, rate)
        a.append(a_)
        b.append(b_)

    return a, b


def test_pca_one_value(name: str, rate: float = 0.9):
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    # set up dictionary of metric constants.
    constants = {
        0.1: [0.839, 0, 1.366, 1.109],
        0.2: [0.882, 0, 1.2, 1.265],
        0.3: [0.963, 0, 1.074, 1.14],
        0.4: [1.027, 0, 1.035, 1.035],
        0.5: [1.026, 0, 1.026, 1.032],
        0.6: [1.026, 0, 1.026, 1.026],
        0.7: [1.026, 0, 0.996, 1.024],
        0.8: [1.024, 0, 0.958, 1.024],
        0.9: [0.981, 0, 1.014, 1.036],
    }

    # remove data.
    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)

    data_true = data_.copy()
    true_train, true_test = train_test_split(data_true, test_size=0.2)
    results = {"no removal": classify.run_xgb(true_train, true_test)[1], "unfilled": 0, "filled": 0}

    data, mask = dp.remove_random_cells(data_, rate)
    x = data.drop(data.columns[-1], axis=1)
    y = data.iloc[:, -1].copy()
    unfilled_data = pd.concat([x, y], axis=1)
    unfilled_train, unfilled_test = train_test_split(unfilled_data, test_size=0.2)
    results["unfilled"] = classify.run_xgb(unfilled_train, unfilled_test)[1]

    # fill the data.
    # step 1- use the pca projection.
    proj_data = run_pca(x)
    # step 2- calculate edges.
    t_t, xor_vals, t_f, f_t, truth_mask = calc_masks(pd.DataFrame(proj_data))
    dists = calc_l2_normal(data.shape[1], t_t, xor_vals, t_f, f_t, truth_mask, np.array(constants[rate]))
    dists = np.sqrt(dists)
    edges = mg.get_knn_edges(dists, 5)

    # step 3- fill the data.
    x = torch.from_numpy(proj_data.astype(np.float32)).to(device)
    y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)
    filled_features = (
        filling("feature_propagation", edges, x, mask, 40, )
    )
    filled_data = pd.DataFrame(filled_features.cpu().numpy())

    # run xgb on filled data.
    filled = pd.concat([filled_data, pd.DataFrame(y)], axis=1)
    filled_train, filled_test = train_test_split(filled, test_size=0.2)
    results["filled"] = classify.run_xgb(filled_train, filled_test)[1]

    # print results.
    for (size, result) in results.items():
        print(f"{size}: {result:.3f}")

    return results["unfilled"], results["filled"]


def test_regression_features(name: str, rate: float = 0.9):
    # remove data.
    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.iloc[:, :-1].copy().values
    y = data_.iloc[:, -1].copy()
    y = y.reset_index(drop=True)

    data, mask = dp.remove_random_cells(data_.copy(), rate)
    x = data.drop(data.columns[-1], axis=1)

    x = x.values

    for i in range(x_.shape[1]):
        # get rows where the feature is not nan.
        rows = ~np.isnan(x[:, i])
        z = np.nan_to_num(x.copy())
        z = np.delete(z, i, axis=1)
        data_i = z[rows]
        labels_i = x[rows, i]
        # train the regression model on the rows.
        reg = LinearRegression()
        reg.fit(data_i, labels_i)

        # fill the missing values with the regression model.
        x[~rows, i] = np.dot(z[~rows], reg.coef_)
        print(np.corrcoef(x_[:, i], x[:, i])[0, 1], end=", ")

    return pd.concat([pd.DataFrame(x), y], axis=1)


def test_regression_features_(data: pd.DataFrame, ):
    """with removed data."""
    # remove data.

    y = data.iloc[:, -1].copy()
    y = y.reset_index(drop=True)

    x = data.drop(data.columns[-1], axis=1)

    x = x.values

    for i in range(x.shape[1]):
        # get rows where the feature is not nan.
        rows = ~np.isnan(x[:, i])
        z = np.nan_to_num(x.copy())
        z = np.delete(z, i, axis=1)
        data_i = z[rows]
        labels_i = x[rows, i]
        # train the regression model on the rows.
        reg = LinearRegression()
        reg.fit(data_i, labels_i)

        # fill the missing values with the regression model.
        x[~rows, i] = np.dot(z[~rows], reg.coef_)

    return pd.concat([pd.DataFrame(x), y], axis=1)


def transform(data: np.array, coefficients: np.array, index: int):
    """
    this method takes a dataset, and a set of coefficients, and returns a transformed dataset.
    :param data: the dataset.
    :param coefficients: the coefficients.
    :param index: the index of the feature to transform.
    :return: the transformed dataset.
    """
    rows = ~np.isnan(data[:, index])
    z = np.nan_to_num(data.copy())
    z = np.delete(z, index, axis=1)
    data[~rows, index] = np.dot(z[~rows], coefficients)
    return data


def plot_regression_results(name: str, rates: list):
    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )
    # remove data.
    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.iloc[:, :-1].copy()

    dists_true = calc_l2(x_)
    edges_true = mg.get_knn_edges(dists_true, 5)
    x_ = x_.values

    # set up dictionary of metric constants.
    constants = {
        0.1: [0.839, 0, 1.366, 1.109],
        0.2: [0.882, 0, 1.2, 1.265],
        0.3: [0.963, 0, 1.074, 1.14],
        0.4: [1.027, 0, 1.035, 1.035],
        0.5: [1.026, 0, 1.026, 1.032],
        0.6: [1.026, 0, 1.026, 1.026],
        0.7: [1.026, 0, 0.996, 1.024],
        0.8: [1.024, 0, 0.958, 1.024],
        0.9: [0.981, 0, 1.014, 1.036],
    }

    a, b = [], []

    for rate in rates:
        print(f"rate: {int(100 * rate)}%")
        data, mask = dp.remove_random_cells(data_.copy(), rate)

        # run xgb on the unfilled and filled data.
        x = data.drop(data.copy().columns[-1], axis=1)
        y = data.iloc[:, -1].copy()
        unfilled_data = pd.concat([pd.DataFrame(x), y], axis=1)
        unfilled_train, unfilled_test = train_test_split(unfilled_data, test_size=0.2)
        a.append(classify.run_xgb(unfilled_train, unfilled_test)[1])

        data = test_regression_features_(data, )
        x = data.drop(data.columns[-1], axis=1)
        print("feature correlations before and after regression:")
        for j in range(x_.shape[1]):
            print(np.corrcoef(x_[:, j], x.iloc[:, j].values)[0, 1], end=", ")
        print()

        # fill data.
        t_t, xor_vals, t_f, f_t, truth_mask = calc_masks(pd.DataFrame(x))
        dists = calc_l2_normal(data.shape[1], t_t, xor_vals, t_f, f_t, truth_mask, np.array(constants[rate]))
        dists = np.sqrt(dists)
        edges = mg.get_knn_edges(dists, 5)

        print("edge intersection: ", len(find_intersection(edges_true.t().numpy(), edges.t().numpy())))
        print("edge union: ", len(find_union(edges_true.t().numpy(), edges.t().numpy())))
        print("% of correct edges: ", len(find_intersection(edges_true.t().numpy(), edges.t().numpy())) / len(
            find_union(edges_true.t().numpy(), edges.t().numpy())))
        x = torch.from_numpy(x.values.astype(np.float32)).to(device)
        y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)
        filled_features = (
            filling("feature_propagation", edges, x, mask, 40, )
        )
        filled_data = pd.DataFrame(filled_features.cpu().numpy())
        filled = pd.concat([filled_data, pd.DataFrame(y)], axis=1)
        filled_train, filled_test = train_test_split(filled, test_size=0.2)
        b.append(classify.run_xgb(filled_train, filled_test)[1])
        print("scores (unfilled and filled):")
        print(a[-1], b[-1])

    return a, b


def regression_metric(data: pd.DataFrame, c: np.array):
    """
    this method assumes the data was filled with linear regression, and calculates the metric on the filled data.
    :param data: the data.
    :param c: penalty constants.
    :return: matrix of distances.
    """
    mask = data.isna().astype(int)  # 1 for nan.

    dists = calc_l2(data)
    xor_dists = calc_l2(pd.DataFrame(mask))
    false_pairs = np.dot(mask, mask.T)

    return dists + c[0] * xor_dists + c[1] * false_pairs


def reg_metric(data: pd.DataFrame, mask: torch.tensor):
    """
    this method calculates the metric on the filled data, assuming it was filled with linear regression.
    :param data: the filled data.
    :param mask: the mask of missing values.
    :return: the metric.
    """
    dists = calc_l2(data)
    xor_dists = calc_l2(pd.DataFrame(mask))
    false_pairs = np.dot(mask, mask.T)

    return dists + xor_dists + 2 * false_pairs


def mink_dist(data: pd.DataFrame, cov: np.array):
    """
    calculates the minkowski distance between all pairs of samples.
    :param data: the dataset.
    :param cov: the covariance matrix.
    :return: matrix of distances.
    :note: this assumes a z-scored, and filled with regression data.
    """
    data = data.copy().values
    inv = np.linalg.inv(cov)
    dists = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            dists[i, j] = dists[j, i] = np.dot(np.dot(data[i] - data[j], inv), data[i] - data[j])

    return dists


def test_minkowski(name: str, rates: list, use_full_data: bool = False):
    """
    this method tests the minkowski distance metric on the filled data.
    :param name: name of the dataset.
    :param rates: rates of missing values.
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    a, b = [], []
    if use_full_data:
        c = []
    for rate in rates:
        print(f"rate: {int(100 * rate)}%")
        data_ = pd.read_csv(f"data/{name}.csv")
        data_ = shuffle(data_)
        data_ = dp.z_score(data_)
        if use_full_data:
            x_ = data_.copy()
            x_ = x_.drop(x_.columns[-1], axis=1)
            data, mask = dp.remove_random_cells(data_.copy(), rate)
            x = data.drop(data.columns[-1], axis=1)
            y = data.iloc[:, -1].copy()
            unfilled_data = pd.concat([x, y], axis=1)
            unfilled_train, unfilled_test = train_test_split(unfilled_data, test_size=0.2)
            a.append(classify.run_xgb(unfilled_train, unfilled_test)[1])

            dists_mink = mink_dist(x_, find_cov_matrix(x_))
            edges = mg.get_knn_edges(dists_mink, 5)
            a_ = edges.t().numpy()

            dists_l2 = calc_l2(x_)
            edges_l2 = mg.get_knn_edges(dists_l2, 5)
            l2df = edges_l2.t().numpy()

            # checking intersections:
            print("% of intersection (mink vs l2 in full data): ",  # how many edges are the same in the correct graph.
                  len(find_intersection(a_, l2df)) / len(find_union(a_, l2df)))

            x = torch.from_numpy(x.values.astype(np.float32)).to(device)
            y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)
            filled_features1 = (
                filling("feature_propagation", edges, x, mask, 40, )
            )
            filled_features2 = (
                filling("feature_propagation", edges_l2, x, mask, 40, )
            )
            filled_data1 = pd.DataFrame(filled_features1.cpu().numpy())
            filled_data2 = pd.DataFrame(filled_features2.cpu().numpy())
            filled1 = pd.concat([filled_data1, pd.DataFrame(y)], axis=1)
            filled2 = pd.concat([filled_data2, pd.DataFrame(y)], axis=1)
            filled_train1, filled_test1 = train_test_split(filled1, test_size=0.2)
            filled_train2, filled_test2 = train_test_split(filled2, test_size=0.2)
            b.append(classify.run_xgb(filled_train1, filled_test1)[1])
            if use_full_data:
                c.append(classify.run_xgb(filled_train2, filled_test2)[1])
            print("scores (unfilled, filled l2, filled mink):")
            print(a[-1], b[-1], c[-1])
            continue

        data, mask = dp.remove_random_cells(data_.copy(), rate)
        data = test_regression_features_(data, )
        x = data.drop(data.columns[-1], axis=1)
        y = data.iloc[:, -1].copy()

        # calculate covariance matrix.
        cov = find_cov_matrix(x)

        # calculate the metric.
        dists = mink_dist(x, cov)

        # run xgb on the unfilled and filled data.
        unfilled_data = pd.concat([x, y], axis=1)
        unfilled_train, unfilled_test = train_test_split(unfilled_data, test_size=0.2)
        a.append(classify.run_xgb(unfilled_train, unfilled_test)[1])

        x = torch.from_numpy(x.values.astype(np.float32)).to(device)
        y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)
        filled_features = (
            filling("feature_propagation", mg.get_knn_edges(dists, 5), x, mask, 40, )
        )
        filled_data = pd.DataFrame(filled_features.cpu().numpy())
        filled = pd.concat([filled_data, pd.DataFrame(y)], axis=1)
        filled_train, filled_test = train_test_split(filled, test_size=0.2)
        b.append(classify.run_xgb(filled_train, filled_test)[1])
        print("scores (unfilled and filled):")
        print(a[-1], b[-1])

    return a, b


def reg_metric_v2(data: pd.DataFrame, mink_dists: np.array, c: np.array):
    """
    this method assumes the data was filled with linear regression, and calculates the metric on the filled data.
    :param mink_dists: minkowski distances.
    :param data: the data.
    :param c: penalty constants.
    :return: matrix of distances.
    """
    mask = data.isna().astype(int)  # 1 for nan.

    dists = mink_dists
    xor_dists = calc_l2(pd.DataFrame(mask))
    false_pairs = np.dot(mask, mask.T)

    return dists + c[0] * xor_dists + c[1] * false_pairs


def plot_cumulative_sum(name: str, rate: float = 0.1):
    """
    this method calculates the true and metric based edges. Then it sorts the metric edges by certainty
    (the ratio between the intersection and total dimensions), and creates a list of flags in the same order,
    when flag is 1 if the edge is in the true graph, and 0 otherwise. Then it calculates the cumulative sum of the
    flags, and plots it.
    :param name: the name of the dataset.
    :param rate: the rate of missing values.
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = mink_dist(x_, find_cov_matrix(x_))

    # calculate true edges.
    edges_true = mg.get_knn_edges(distances=distances_full, k=5)
    # keep only uniques.
    edges_true = edges_true.t().numpy()
    edges_true = np.unique(edges_true, axis=0)

    # remove data.
    data, mask = dp.remove_random_cells(data_.copy(), rate)

    # filling with regression first.
    x = test_regression_features_(data, )
    x = x.drop(x.columns[-1], axis=1)
    # calculate metric distances.
    dists = mink_dist(x, find_cov_matrix(x))
    edges = mg.get_knn_edges(dists, 5)
    edges = edges.t().numpy()
    edges = np.unique(edges, axis=0)

    # sorting edges by certainty.
    certainties = np.zeros(edges.shape[0])
    for i, edge in enumerate(edges):
        xi = mask[edge[0]].numpy().astype(int)
        xj = mask[edge[1]].numpy().astype(int)
        certainties[i] = np.dot(xi, xj) / len(xi)

    # making a temp array with the edges and certainties.

    temp = np.append(edges, certainties.T.reshape(-1, 1), axis=1)

    # creating flags.
    flags = np.zeros(temp.shape[0])
    for i in range(temp.shape[0]):
        flags[i] = int(np.any(np.all(edges_true == temp[i, :2], axis=1)))

    flags = np.array(flags)
    temp = np.append(temp, flags.T.reshape(-1, 1), axis=1)

    # sorting the temp array by the certainties.
    indices = np.argsort(temp[:, 2])[::-1]
    temp = temp[indices]

    flags = temp[:, -1]

    print(sum(flags))
    print(len(find_intersection(edges_true, edges)))
    # plotting cumulative sum.
    plt.plot(np.cumsum(flags))
    plt.title(f"Cumulative sum of correct edges for {name} at {rate * 100}%")
    plt.xlabel("Edge number")
    plt.ylabel("Cumulative sum")
    plt.grid()
    plt.savefig(f"plots/cumulative_sum_{name}_{int(rate * 100)}.png")
    plt.show()


def test_certain_edges_(name: str, rate: float = 0.1):
    """
    this method calculates the true and metric based edges. Then it sorts the metric edges by certainty
    (the ratio between the intersection and total dimensions), and creates a list of flags in the same order,
    when flag is 1 if the edge is in the true graph, and 0 otherwise. Then it calculates the cumulative sum of the
    flags, and plots it.
    :param name: the name of the dataset.
    :param rate: the rate of missing values.
    :return:
    """

    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu"
    )

    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = mink_dist(x_, find_cov_matrix(x_))

    # calculate true edges.
    edges_true = mg.get_knn_edges(distances=distances_full, k=5)
    # keep only uniques.
    edges_true = edges_true.t().numpy()

    print("total true edges:", edges_true.shape)

    edges_true = np.unique(edges_true, axis=0)

    print("total unique true edges:", edges_true.shape)

    original = data_.copy()

    # remove data.
    data, mask = dp.remove_random_cells(data_.copy(), rate)
    unfilled = data.copy()

    # filling with regression first.
    x = test_regression_features_(data, )
    x = x.drop(x.columns[-1], axis=1)
    # calculate metric distances.
    dists = mink_dist(x, find_cov_matrix(x))
    edges = mg.get_knn_edges(dists, 5)
    edges = edges.t().numpy()

    print("total metric edges:", edges.shape)

    edges = np.unique(edges, axis=0)

    # sorting edges by certainty.
    certainties = np.zeros(edges.shape[0])
    for i, edge in enumerate(edges):
        xi = mask[edge[0]].numpy().astype(int)
        xj = mask[edge[1]].numpy().astype(int)
        certainties[i] = np.dot(xi, xj) / len(xi)

    # making a temp array with the edges and certainties.

    temp = np.append(edges, certainties.T.reshape(-1, 1), axis=1)

    # creating flags.
    flags = np.zeros(temp.shape[0])
    for i in range(temp.shape[0]):
        flags[i] = int(np.any(np.all(edges_true == temp[i, :2], axis=1)))

    flags = np.array(flags)
    temp = np.append(temp, flags.T.reshape(-1, 1), axis=1)

    # sorting the temp array by the certainties.
    indices = np.argsort(temp[:, 2])[::-1]
    temp = temp[indices]

    # take only top percentage of edges.
    temp = temp[temp[:, 2] > 0]
    edges = temp[: temp.shape[0] // 2, :2]

    print("total unique certain edges", edges.shape)

    edges = torch.from_numpy(edges.astype(np.int64)).t()

    # fill the data.
    x = torch.from_numpy(x.values.astype(np.float32)).to(device)
    y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)

    filled_features = (
        filling("feature_propagation", edges, x, mask, 40, )
    )
    filled_data = pd.DataFrame(filled_features.cpu().numpy())
    filled = pd.concat([filled_data, pd.DataFrame(y)], axis=1)

    # run xgb on filled data.
    o_train, o_test = train_test_split(original, test_size=0.2)
    u_train, u_test = train_test_split(unfilled, test_size=0.2)
    f_train, f_test = train_test_split(filled, test_size=0.2)

    _, a = classify.run_xgb(o_train, o_test)
    _, b = classify.run_xgb(u_train, u_test)
    _, c = classify.run_xgb(f_train, f_test)

    return a, b, c


def test_certain_edges(name: str, rates: list):  # todo- finish this method.
    """
    this method is a wrapper for the test_certain_edges_ method. for each rate it calculates the true and metric edges,
    and keeps only the most certain edges. then it runs xgb on the unfilled and filled data.
    :param name: the dataset
    :param rates: rates of missing values.
    :return:
    """
    a, b, c = [], [], []
    for rate in rates:
        print(f"rate: {int(100 * rate)}%")
        a_, b_, c_ = test_certain_edges_(name, rate)
        a.append(a_)
        b.append(b_)
        c.append(c_)

    return a, b, c


def plot_k_vals(name: str, rates: list, k_vals: list):
    """
    this method plots the xgb scores for the full data for different k values.
    :param name: the dataset name.
    :param rates: rates of missing values.
    :param k_vals: the k values.
    :return:
    """
    results_true = {}  # only true edges.
    results_reg = {}  # only metric edges.
    results_inter = {}  # only intersection.
    for k in k_vals:
        results_true[k] = []
        results_reg[k] = []
        results_inter[k] = []

    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu"
    )

    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = mink_dist(x_, find_cov_matrix(x_))

    for k in k_vals:
        print(f"k: {k}")
        # calculate true edges.
        edges_true = mg.get_knn_edges(distances=distances_full, k=k)
        # keep only uniques.
        edges_true = edges_true.t().numpy()
        edges_true = np.unique(edges_true, axis=0)
        edges_true = torch.from_numpy(edges_true.astype(np.int64)).t()

        for rate in rates:
            print(f"rate: {int(100 * rate)}%")
            # remove data.
            data, mask = dp.remove_random_cells(data_.copy(), rate)
            data = z_score(data)
            x = data.drop(data.columns[-1], axis=1)
            y = data.iloc[:, -1].copy()

            # fill with regression.
            x_reg = test_regression_features_(data, )
            x_reg = x_reg.drop(x_reg.columns[-1], axis=1)

            # calculate metric distances.
            dists = mink_dist(x_reg, find_cov_matrix(x_reg))
            edges = mg.get_knn_edges(dists, k).t().numpy()
            edges = np.unique(edges, axis=0)
            edges = torch.from_numpy(edges.astype(np.int64)).t()

            # calculate intersection.
            inter = find_intersection(edges_true.t().numpy(), edges.t().numpy())
            inter = torch.from_numpy(np.array(list(inter))).t()

            # fill the data.
            x = torch.from_numpy(x.values.astype(np.float32)).to(device)
            y = torch.from_numpy(y.values.astype(np.int64)).to(device)
            x_reg = torch.from_numpy(x_reg.values.astype(np.float32)).to(device)

            filling_all = filling("feature_propagation", edges_true, x, mask, 40, )
            filling_reg = filling("feature_propagation", edges, x_reg, mask, 40, )
            filling_inter = filling("feature_propagation", inter, x_reg, mask, 40, )

            filled_all = pd.concat([pd.DataFrame(filling_all.cpu().numpy()), pd.DataFrame(y)], axis=1)
            filled_reg = pd.concat([pd.DataFrame(filling_reg.cpu().numpy()), pd.DataFrame(y)], axis=1)
            filled_inter = pd.concat([pd.DataFrame(filling_inter.cpu().numpy()), pd.DataFrame(y)], axis=1)

            # run xgb on filled data.
            train_all, test_all = train_test_split(filled_all, test_size=0.2)
            train_reg, test_reg = train_test_split(filled_reg, test_size=0.2)
            train_inter, test_inter = train_test_split(filled_inter, test_size=0.2)

            results_true[k].append(classify.run_xgb(train_all, test_all)[1])
            results_reg[k].append(classify.run_xgb(train_reg, test_reg)[1])
            results_inter[k].append(classify.run_xgb(train_inter, test_inter)[1])

    # plot results.
    for k in k_vals:
        plt.plot(rates, results_true[k], label=f"true edges, k={k}")
    plt.xlabel("Rate of missing values")
    plt.ylabel("AUC score")
    plt.legend()
    plt.grid()
    plt.title(f"FP with true edges for different K values in {name}")
    plt.savefig(f"plots/k_vals_true_{name}.png")
    plt.show()
    plt.clf()

    for k in k_vals:
        plt.plot(rates, results_reg[k], label=f"metric edges, k={k}")
    plt.xlabel("Rate of missing values")
    plt.ylabel("AUC score")
    plt.legend()
    plt.grid()
    plt.title(f"FP with metric edges for different K values in {name}")
    plt.savefig(f"plots/k_vals_reg_{name}.png")
    plt.show()
    plt.clf()

    for k in k_vals:
        plt.plot(rates, results_inter[k], label=f"intersection, k={k}")
    plt.xlabel("Rate of missing values")
    plt.ylabel("AUC score")
    plt.legend()
    plt.grid()
    plt.title(f"FP with intersection for different K values in {name}")
    plt.savefig(f"plots/k_vals_inter_{name}.png")
    plt.show()
    plt.clf()


def plot_results_intersection(name: str, rates: list, k=40):
    """
    this method calculates the rate of intersection between the true and metric edges (noted as p),
     and uses the p smallest distances among the knn graph as the edges for FP.
    :param name:
    :param rates:
    :param k:
    :return:
    """

    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu"
    )

    results = []

    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = mink_dist(x_, find_cov_matrix(x_))
    # calculate true edges.
    edges_true = mg.get_knn_edges(distances=distances_full, k=k)
    # keep only uniques.
    edges_true = edges_true.t().numpy()
    edges_true = np.unique(edges_true, axis=0)
    edges_true = torch.from_numpy(edges_true.astype(np.int64)).t()

    for rate in rates:
        print(f"rate: {int(100 * rate)}%")
        # remove data.
        data, mask = dp.remove_random_cells(data_.copy(), rate)
        data = z_score(data)
        y = data.iloc[:, -1].copy()

        # fill with regression.
        x_reg = test_regression_features_(data, )
        x_reg = x_reg.drop(x_reg.columns[-1], axis=1)

        # calculate metric distances.
        dists = mink_dist(x_reg, find_cov_matrix(x_reg))
        edges = mg.get_knn_edges(dists, k).t().numpy()
        edges = np.unique(edges, axis=0)
        edges = torch.from_numpy(edges.astype(np.int64)).t()

        # calculate intersection in percentage.
        p = len(find_intersection(edges_true.t().numpy(), edges.t().numpy())) / edges.shape[1]
        print(f"intersection percentage: {p * 100}%")

        # sort the metric distances from smallest to largest.
        dists = [dists[edge[0], edge[1]] for edge in edges.t().numpy()]
        dists = np.array(dists).reshape(-1, 1)

        temp = np.concatenate((edges.t().numpy(), dists), axis=1)
        temp = temp[temp[:, -1].argsort()]

        # take the p smallest distances.
        edges = temp[:int(p * edges.shape[1]), :2]
        edges = torch.from_numpy(edges.astype(np.int64)).t()

        # fill the data.
        x_reg = torch.from_numpy(x_reg.values.astype(np.float32)).to(device)
        y = torch.from_numpy(y.values.astype(np.int64)).to(device)

        filling_reg = filling("feature_propagation", edges, x_reg, mask, 40, )
        filled_reg = pd.concat([pd.DataFrame(filling_reg.cpu().numpy()), pd.DataFrame(y)], axis=1)

        # run xgb on filled data.
        train_reg, test_reg = train_test_split(filled_reg, test_size=0.2)

        results.append(classify.run_xgb(train_reg, test_reg)[1])

    plt.plot(rates, results, )
    plt.xlabel("Rate of missing values")
    plt.ylabel("AUC score")
    plt.legend()
    plt.grid()
    plt.title(f"FP with intersection for different rates in {name}")
    plt.savefig(f"plots/intersection_{name}.png")
    plt.show()


def test_confident_neighbors(name: str, rates: list):
    """
    this method creates the knn(40) graph for the missing data after regression, then for each node, it keeps a
    percentage of its neighbors with the highest confidence.
    :param name:
    :param rates:
    :return:
    """
    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu"
    )
    results = {}

    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.copy().drop(data_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)
    # use knn method to find the nodes between which the distances are smallest.
    edges_true = mg.get_knn_edges(distances=distances_full, k=40)
    a = edges_true.t().numpy()

    for rate in rates:
        print(f"rate: {int(100 * rate)}%")
        # remove data.
        data, mask = dp.remove_random_cells(data_.copy(), rate)
        data = z_score(data)
        unfilled = data.copy()
        y = data.iloc[:, -1].copy()
        x = test_regression_features_(data, )
        x = x.drop(x.columns[-1], axis=1)
        # calculate metric distances.
        dists = calc_l2(x)
        edges = mg.get_knn_edges(dists, 40).t().numpy()
        edges = np.unique(edges, axis=0)
        # for each node, keep the c% of its neighbors with the highest confidence.
        confidence = len(find_intersection(a, edges))
        u = edges.shape[0]
        confidence = confidence / u

        print("Total edges: ", edges.shape[0])

        print(f"intersection percentage: {confidence:.3f}")
        print("edge intersection: ", len(find_intersection(a, edges)))

        # for each node, keep confidence 5 of its neighbors.

        dists_ = np.array([dists[edge[0], edge[1]] for edge in edges])
        temp = np.concatenate((edges, dists_.reshape(-1, 1)), axis=1)

        temp = temp[temp[:, -1].argsort()]

        temp = temp[:, :-1].astype(int)
        print("-----------------------------------")
        # for each node, keep the closest 'confidence' 5 of its neighbors.
        neighbors = []
        for node in temp[:, 0].astype(int):
            neighbors_n = temp[temp[:, 0] == node]
            # we know that the smallest are the lowest instances.
            neighbors_n = neighbors_n[:int(confidence * neighbors_n.shape[0])]
            neighbors.extend(neighbors_n)

        neighbors = np.unique(np.array(neighbors), axis=0)
        neighbors = torch.from_numpy(np.array(neighbors).astype(np.int64)).t()

        print("Total neighbors: ", neighbors.shape[1])

        # check the intersection.
        print("edge intersection after removal: ", len(find_intersection(a, neighbors.t().numpy())))
        u = neighbors.shape[1]
        conf_after = len(find_intersection(a, neighbors.t().numpy())) / u
        print(f"intersection percentage after removal: {conf_after:.3f}")
        print("-----------------------------------")

        # fill the data.
        x = torch.from_numpy(x.values.astype(np.float32)).to(device)
        y = torch.from_numpy(y.values.astype(np.int64)).to(device)
        filling_reg = filling("feature_propagation", neighbors, x, mask, 40, )
        filled_reg = pd.concat([pd.DataFrame(filling_reg.cpu().numpy()), pd.DataFrame(y)], axis=1)
        # run xgb on unfilled and filled data.
        unfilled_train, unfilled_test = train_test_split(unfilled, test_size=0.2)
        train_reg, test_reg = train_test_split(filled_reg, test_size=0.2)
        results[rate] = (round(classify.run_xgb(unfilled_train, unfilled_test)[1], 3),
                         round(classify.run_xgb(train_reg, test_reg)[1], 3), round(confidence, 3), round(conf_after, 3))

    # plot results.
    plt.plot(rates, [results[rate][0] for rate in rates], label="unfilled")
    plt.plot(rates, [results[rate][1] for rate in rates], label="filled")
    plt.xlabel("Rate of missing values")
    plt.ylabel("AUC score")
    plt.grid()
    plt.legend()
    plt.title(f"FP with confident neighbors for different rates in {name}")
    plt.savefig(f"plots/confidence_{name}_temp.png")
    plt.show()
    # make a dictionary with results: rate, auc, confidence.
    dct = pd.DataFrame(results).T
    # give the columns names.
    dct.columns = ["AUC before filling", "AUC after filling", "Intersection percentage before",
                   "Intersection percentage after"]
    dct.to_csv(f"results_confidence_{name}.csv")


def plot_scatter_confidence(name: str, rates: list, k=40):
    data_ = pd.read_csv(f"data/{name}.csv")

    # shuffle.
    data_ = shuffle(data_)
    # z-score.
    data_ = dp.z_score(data_)

    x_ = data_.copy()
    x_ = x_.drop(x_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)
    # use knn method to find the nodes between which the distances are smallest.
    edges_true = mg.get_knn_edges(distances=distances_full, k=k)
    a = edges_true.t().numpy()

    # for each rate, calculate distances.
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, rate in enumerate(rates):
        # remove data.
        print(f"rate: {rate}")
        data, mask = dp.remove_random_cells(data_.copy(), rate)

        # remove rows if all values are nan.
        data = data.dropna(axis=0, how="all")
        print(data.shape)

        data = test_regression_features_(data)
        data = data.drop(data.columns[-1], axis=1)

        dists = reg_metric(data, mask.cpu().detach().numpy().astype(int))

        # calculate edges.
        edges = mg.get_knn_edges(dists, k)
        b = edges.t().numpy()

        # calculates confidence for each edge.
        confidences = np.zeros(b.shape[0])
        for j, edge in enumerate(b):
            xi = mask[edge[0]].numpy().astype(int)
            xj = mask[edge[1]].numpy().astype(int)
            confidences[j] = np.dot(xi, xj) / len(xi)

        temp = np.concatenate((b, confidences.T.reshape(-1, 1)), axis=1)
        conf = temp[temp[:, -1] > 0.5, :2].astype(int)
        unconf = temp[temp[:, -1] <= 0.5, :2].astype(int)

        # find the intersection and set differences between the edges.
        inter = np.array(list(find_intersection(a, b)))

        # save the distances according to the presence of the edges.
        conf_x_in = [distances_full[x[0], x[1]] for x in conf if x in inter]
        conf_y_in = [dists[x[0], x[1]] for x in conf if x in inter]
        conf_x_out = [distances_full[x[0], x[1]] for x in conf if x not in inter]
        conf_y_out = [dists[x[0], x[1]] for x in conf if x not in inter]
        unconf_x_in = [distances_full[x[0], x[1]] for x in unconf if x in inter]
        unconf_y_in = [dists[x[0], x[1]] for x in unconf if x in inter]
        unconf_x_out = [distances_full[x[0], x[1]] for x in unconf if x not in inter]
        unconf_y_out = [dists[x[0], x[1]] for x in unconf if x not in inter]

        # plot the scatters.
        axs[i // 3, i % 3].scatter(unconf_x_in, unconf_y_in, c="red", label="Unconfident in intersection", marker="^")
        axs[i // 3, i % 3].scatter(unconf_x_out, unconf_y_out, c="blue", label="Unconfident not in intersection",
                                   marker="^")
        axs[i // 3, i % 3].scatter(conf_x_in, conf_y_in, c="green", label="Confident in intersection", marker="o")
        axs[i // 3, i % 3].scatter(conf_x_out, conf_y_out, c="black", label="Confident not in intersection", marker="x")
        axs[i // 3, i % 3].legend(loc="upper right")
        axs[i // 3, i % 3].set_title(f"Rate: {rate}, K={k}, Intersection: {len(inter)}")
        axs[i // 3, i % 3].set_xlabel("Original difference")
        axs[i // 3, i % 3].set_ylabel("Metric difference")
        axs[i // 3, i % 3].grid()

    plt.savefig(f"plots/scatter_{name}_confidence.png")
    fig.show()


def metric_by_feature(data: pd.DataFrame, idx: int):
    """
    this method calculates the metric for each feature, assuming it was filled with linear regression.
    :param data: the dataset.
    :param idx: the index of the feature.
    :return: the metric.
    """
    _data = data.copy().values
    valid_indices = ~np.isnan(_data[:, idx])
    valid_data = _data[valid_indices, idx]

    diffs = np.subtract.outer(valid_data, valid_data) ** 2
    np.fill_diagonal(diffs, np.nan)  # Diagonal should be NaNs
    max_val = np.nanmax(diffs)
    diffs[np.isnan(diffs)] = max_val  # Replace NaNs with max value

    dists = np.empty((_data.shape[0], _data.shape[0]))
    dists.fill(max_val)  # Fill with max_val initially
    dists[np.ix_(valid_indices, valid_indices)] = diffs

    return dists


def test_knn_foreach_feature_(name: str, rate: float, ):
    """
    this method collects edges for the final graph using the following method:
    for each feature, build a knn graph with k=40 on the UNFILLED, then take the union of all the edges.
    :param name: name of the dataset.
    :param rate: missing values rate.
    :return: (unfilled auc, filled auc, intersection percentage)
    """
    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu"
    )

    data_ = pd.read_csv(f"data/{name}.csv")
    data_ = shuffle(data_)
    data_ = dp.z_score(data_)
    x_ = data_.copy().drop(data_.columns[-1], axis=1)

    # calculate distances on full data.
    distances_full = calc_l2(x_)
    # use knn method to find the nodes between which the distances are smallest.
    edges_true = mg.get_knn_edges(distances=distances_full, k=40)
    a = edges_true.t().numpy()
    a = np.unique(a, axis=0)

    # remove data.
    data, mask = dp.remove_random_cells(data_.copy(), rate)
    data = z_score(data)
    unfilled = data.copy()

    x = data.drop(data.columns[-1], axis=1)
    # collect edges for unfilled data graph.
    edges = []
    for i in range(x.shape[1]):
        dists = metric_by_feature(x, i)
        edges.extend(mg.get_knn_edges(dists, 10).t().numpy().tolist())

    edges = np.unique(np.array(edges), axis=0)

    x_reg = test_regression_features_(data.copy(), )
    dists_ = calc_l2(x_reg)

    mask_vals = ~mask.cpu().numpy().astype(int)

    dists_ = calc_l2(x_reg)
    edges_l2 = mg.get_knn_edges(dists_, 40)

    # intersect edges and edges_l2.
    edges_l2 = edges_l2.t().numpy()
    edges_l2 = np.unique(edges_l2, axis=0)
    edges = np.unique(edges, axis=0)
    edges = np.array(list(find_intersection(edges, edges_l2)))

    # calculate intersection.
    perc = len(find_intersection(a, edges)) / edges.shape[0]

    edges = torch.from_numpy(edges.astype(np.int64)).t()

    # fill the data.
    x = torch.from_numpy(x.values.astype(np.float32)).to(device)
    y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)
    filling_reg = filling("feature_propagation", edges, x, mask, 40, )
    filled_reg = pd.concat([pd.DataFrame(filling_reg.cpu().numpy()), pd.DataFrame(y)], axis=1)
    # run xgb on unfilled and filled data.
    unfilled_train, unfilled_test = train_test_split(unfilled, test_size=0.2)
    train_reg, test_reg = train_test_split(filled_reg, test_size=0.2)

    _, a = classify.run_xgb(unfilled_train, unfilled_test)
    _, b = classify.run_xgb(train_reg, test_reg)

    return a, b, perc


def test_knn_foreach_feature(name: str, rates: list, iters: int = 25):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dct = {}

    df = pd.read_csv(f"data/{name}.csv")
    df = shuffle(df)
    df = dp.z_score(df)
    # add a straight line of the full data auc.
    full_train, full_test = train_test_split(df, test_size=0.2)
    _, auc = classify.run_xgb(full_train, full_test)

    # getting the true edges.
    a = df.copy()
    a = a.drop(a.columns[-1], axis=1)
    distances_full = calc_l2(a)
    edges_true = mg.get_knn_edges(distances=distances_full, k=40)
    fill_scores = {}
    for rate in rates:
        print(f"rate: {int(100 * rate)}%")

        x, y, z = [], [], []
        fill_scores[rate] = []
        # run the test for each rate multiple times.
        for i in range(iters):
            print(f"iteration: {i + 1}")

            a_, b_, c_ = test_knn_foreach_feature_(name, rate)
            x.append(a_)
            y.append(b_)
            z.append(c_)

            # running FP and xgb with the true edges as well.
            data, mask = dp.remove_random_cells(df.copy(), rate)
            data = z_score(data)
            unfilled_x = data.copy().drop(data.columns[-1], axis=1)
            y_ = data.iloc[:, -1].copy()

            unfilled_x = torch.from_numpy(unfilled_x.values.astype(np.float32)).to(device)
            y_ = torch.from_numpy(y_.values.astype(np.int64)).to(device)

            # Filling the data using the true edges.
            filling_true = filling("feature_propagation", edges_true, unfilled_x, mask, 40, )

            filled_true = pd.concat([pd.DataFrame(filling_true.cpu().numpy()), pd.DataFrame(y_)], axis=1)
            train_true, test_true = train_test_split(filled_true, test_size=0.2)
            _, d = classify.run_xgb(train_true, test_true)
            fill_scores[rate].append(d)

        print(f"rate: {rate}, average confidence percentage: {np.mean(z):.3f}+-{np.std(z):.3f}")
        dct[rate] = (x, y, z)

    # for each rate of missing values, plot the average and std of the iteration results for both the unfilled and
    # filled data.
    avs0 = [np.mean(dct[rate][0]) for rate in rates]
    stds0 = [np.std(dct[rate][0]) / np.sqrt(iters) for rate in rates]
    avs1 = [np.mean(dct[rate][1]) for rate in rates]
    stds1 = [np.std(dct[rate][1]) / np.sqrt(iters) for rate in rates]

    fill_means = [np.mean(fill_scores[rate]) for rate in rates]
    fill_stds = [np.std(fill_scores[rate]) / np.sqrt(iters) for rate in rates]

    plt.errorbar(rates, avs0, stds0, label="Unfilled")
    plt.errorbar(rates, avs1, stds1, label="Filled with Metric edges")
    plt.errorbar(rates, fill_means, fill_stds, label="Filled with True edges")

    plt.axhline(y=auc, color="r", linestyle="--", label="Full data")

    plt.xlabel("Rate of missing values")
    # if 2 classes label the y-axis to AUC, else to accuracy.
    if len(df.iloc[:, -1].unique()) == 2:
        plt.ylabel("AUC score")
    else:
        plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.title(f"FP with knn for each feature in {name}")
    plt.savefig(f"table_to_graph_plots/knn_foreach_{name}.png")
    plt.show()
    plt.clf()

    ans = {}
    for rate in rates:
        ans[rate] = (f"{np.mean(dct[rate][0]):.3f}+-{np.std(dct[rate][0]):.3f}",
                     f"{np.mean(dct[rate][1]):.3f}+-{np.std(dct[rate][1]):.3f}",
                     f"{np.mean(dct[rate][2]):.3f}+-{np.std(dct[rate][2]):.3f}")

    # save the results to a csv file.
    dct = pd.DataFrame(ans).T
    dct.columns = ["AUC before filling", "AUC after filling", "Intersection percentage"]
    dct.to_csv(f"table_to_graph_results/knn_foreach_{name}.csv")


def fixed_metric(distances: np.array, mask: np.array, c: np.array):
    """
    this method adds penalties on missing features.
    :param c: penalties.
    :param distances: the original distances after linear regression.
    :param mask: mask of present and missing values. 1 for missing, 0 for present.
    :return:
    """

    dists = np.square(distances)

    for i in range(dists.shape[0]):
        for j in range(i, dists.shape[0]):
            dists[i, j] = dists[j, i] = np.sqrt(dists[i, j] + c[0] * np.sum(mask[i]) + c[1] * np.sum(mask[j]))

    return dists
