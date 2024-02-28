from src import build_classifier as classify
import data_preparation as dp
from fp_builds.filling_strategies import filling
from fp_builds import make_graph as mg
# from sklearn.utils import shuffle
from src.data_filler import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def test_data(name: str):
    """
    shows histogram of the data.
    :param name:
    :return:
    """
    data_ = pd.read_csv(f"data/{name}.csv")
    data = data_.copy()
    for i in range(data.shape[1]):
        x = data.iloc[:, i]
        print(np.median(x), np.mean(x), np.std(x))
        plt.hist(x, 100)
        plt.ylabel("Frequency in column " + str(i))
        plt.show()
        plt.clf()


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


def calc_l2(data: pd.DataFrame):
    """
    This method calculates the L2 distances between rows in the data when NaN values are replaced with 0.
    :param data: the data.
    :return: matrix of distances.
    """
    data = data.fillna(0)
    dists = np.linalg.norm(data.values[:, np.newaxis, :] - data.values[np.newaxis, :, :], axis=-1)

    return dists


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
