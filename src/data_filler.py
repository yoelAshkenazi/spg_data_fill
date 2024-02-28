"""This code is designed to take a dataset (either csv or graph), remove a percentage of the data, and then fill the
missing values using a graph-based method. Note: In order to use this code properly, you need to have the datasets in
the correct path. Note II: currently, the code works for a graph only if the graph is saved in one of the formats
specified in get_dataset.
data_path: data/file_name"""
import pandas as pd
import torch
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid, MixHopSyntheticDataset
import torch_geometric.transforms as transforms
from torch_geometric.typing import Adj
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_scatter import scatter_add
from xgboost import XGBClassifier

GRAPH_DATA_NAMES = ["Cora", "CiteSeer", "PubMed", "OGBN-Arxiv", "OGBN-Products", "MixHopSynthetic"]
# todo- add names of graph datasets.
PATH = "data"


# z-score each column.
def z_score(data_frame: pd.DataFrame) -> pd.DataFrame:
    """

    :param data_frame: DataFrame of the dataset.
    :return: the dataset with z-scored columns.
    """
    for i in range(data_frame.shape[1] - 1):
        column_mean = data_frame.iloc[:, i].mean(skipna=True)
        column_std = data_frame.iloc[:, i].std(skipna=True)

        if column_std != 0 and not np.isnan(column_std):  # Avoid division by zero when std is zero
            data_frame.iloc[:, i] = (data_frame.iloc[:, i] - column_mean) / column_std

    return data_frame


# remove percentage of random cells by replacing them with 0.
def remove_random_cells(data_frame: pd.DataFrame, percentage: float):
    """
    This method takes a dataset and a percentage, and removes the percentage of the data by replacing the cells with 0,
    Then it normalizes the data using z-score.
    :param data_frame: the dataset.
    :param percentage: the rate of missing values.
    :return: the dataset with missing values, and the mask as a torch tensor of boolean values.
    """
    mask = np.ones(data_frame.iloc[:, : -1].shape)
    for i in range(0, data_frame.shape[0]):
        for j in range(0, data_frame.shape[1] - 1):
            if np.random.rand() < percentage:
                mask[i, j] = 0
                data_frame.iloc[i, j] = float("nan")
    data_frame = z_score(data_frame)

    # convert mask to tensor.
    mask = torch.from_numpy(mask.astype(np.float32)).bool()

    return data_frame, mask


# the methods below take care of the data removal in a graph dataset.
def get_dataset(name: str, use_lcc: bool = True, homophily=None):
    """

    :param name: the name of the graph dataset.
    :param use_lcc: whether to use the largest connected component of the graph.
    :param homophily: whether to assume homophily in the graph.
    :return: the dataset and evaluator.
    """
    path = os.path.join("data", name)  # TODO: change this to the correct path.
    evaluator = None

    # if tabular.
    if name not in GRAPH_DATA_NAMES:
        return pd.read_csv(f"data/{name}.csv")

    # if in graph list (known graphs).
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(path, name)
    elif name in ["OGBN-Arxiv", "OGBN-Products"]:
        dataset = PygNodePropPredDataset(name=name.lower(), transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name=name.lower())
        use_lcc = False
    elif name == "MixHopSynthetic":
        dataset = MixHopSyntheticDataset(path, homophily=homophily)
    else:  # a graph but not from the list, or added to the list later.
        # extract the data file and the edges file. both should be in csv format and under "data/name/" directory.
        data_file = os.path.join(path, f"data.csv")  # table of samples and features (rows and columns).
        edges_file = os.path.join(path, f"edges.csv")  # table of edges (2 x number of edges).
        # read the data and the edges.
        data = pd.read_csv(data_file).values.astype(np.float32)
        edges = pd.read_csv(edges_file).values.astype(np.int64)
        # save to a graph in torch.geometric.data.Data format.
        data = Data(
            x=torch.tensor(data[:, :-1]),
            edge_index=torch.tensor(edges),
            y=torch.tensor(data[:, -1]),
        )
        return data

    if use_lcc:
        dataset = keep_only_largest_connected_component(dataset)

    # Make graph undirected so that we have edges for both directions and add self loops
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index, num_nodes=dataset.data.x.shape[0])

    return dataset, evaluator


# get a connected component of the graph.
def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    """

    :param dataset: the dataset.
    :param start: the starting node.
    :return: the connected component of the graph that includes the starting node.
    """
    visited_nodes = set()
    queued_nodes = {start}
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


# get the largest connected component of the graph.
def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    """

    :param dataset: the dataset.
    :return: the largest connected component of the graph.
    """
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


# get the largest connected component of the graph.
def keep_only_largest_connected_component(dataset):
    """

    :param dataset: the dataset.
    :return: the largest connected component of the graph.
    """
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
    )
    dataset.data = data

    return dataset


# take care of the data removal in a graph dataset.
def get_missing_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """
    Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing.
    If `type`='uniform', then each feature of each node is missing uniformly at random with probability `rate`.
    Instead, if `type`='structural', either we observe all features for a node, or we observe none. For each node
    there is a probability of `rate` of not observing any feature.
    """
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()


# this method gets a dataset with missing values, and returns the mask.
def find_mask(dataset: pd.DataFrame) -> torch.Tensor:
    """

    :param dataset: the dataset.
    :return: the mask.
    note: the dataset itself is always a Pandas.DataFrame. the mask is a torch.Tensor.

    """
    # find nan values.
    z = dataset.notna().astype(int)  # 1 for numeric values, 0 for nan values.
    mask = torch.from_numpy(z.values).bool()  # True for numeric values, False for nan values.
    return mask


# FP methods.
class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor) -> Tensor:
        # out is initialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(edge_index, n_nodes)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, edge_index, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted.

        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """

    :param edge_index: a tensor of shape [2, num_edges] containing the indices of the edges in the graph.
    :param n_nodes: the number of nodes in the graph (rows in dataset).
    :return: the symmetrically normalized adjacency matrix of the graph, noted as D^-1/2 * A * D^-1/2.
    :return: the propagation matrix, and the weights of that matrix.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


def calc_l2(data: pd.DataFrame):
    """
    This method calculates the L2 distances between rows in the data when NaN values are replaced with 0.
    :param data: the data.
    :return: matrix of distances.
    """
    data = data.fillna(0)
    dists = np.linalg.norm(data.values[:, np.newaxis, :] - data.values[np.newaxis, :, :], axis=-1)

    return dists


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


def lin_reg(data: pd.DataFrame):
    """
    This method takes a dataset with missing values, and fills the missing values using a linear regression model.
    :param: data: the dataset.
    :return: dataframe with filled values.
    """

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
        x[~rows, i] += reg.intercept_

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


def make_feature_edges(data: pd.DataFrame):
    """
    This method takes a dataset with missing values, and using the metric_by_feature method, it creates a "knn" graph
    for each feature, then intersect it with the knn graph of the linear regression filled data to get the edges.
    :param data: the dataset.
    :return:
    """

    # separate the features from the labels.
    x = data.drop(data.columns[-1], axis=1)

    edges = []
    for i in range(x.shape[1]):
        dists = metric_by_feature(x, i)
        knn_edges = get_knn_edges(dists, k=10).t().numpy().tolist()
        edges.extend(knn_edges)

    edges = np.unique(edges, axis=0)
    x_reg = lin_reg(data.copy(), ).iloc[:, :-1]

    dists_ = calc_l2(x_reg)
    edges_l2 = get_knn_edges(dists_, 40)

    # intersect edges and edges_l2.
    edges_l2 = edges_l2.t().numpy()
    edges_l2 = np.unique(edges_l2, axis=0)
    edges = np.unique(edges, axis=0)
    edges = np.array(list(find_intersection(edges, edges_l2)))

    edges = torch.tensor(edges).t().long()

    return edges


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


def get_knn_edges(distances, k: int = 5):
    """
    this method gets the distances between each row in the data, and returns the tensor of knn edges.
    :param distances: distances between each row in the data.
    :param k: number of neighbors. (defaulted to 5 for connectivity)
    :return: edges of knn graph of the data.
    """
    edges = np.argsort(distances, axis=1)[:, 1:k + 1]  # start with 1 to avoid self edges.
    edges_ = np.column_stack((np.repeat(np.arange(edges.shape[0]), k), edges.flatten()))
    edges = np.vstack((edges_, edges_[:, [1, 0]]))  # Include [j, i] for every [i, j]
    edges = torch.tensor(edges).t().contiguous()
    return edges


# fill missing data using FP.
def fill_data(data_name):
    """
    this method takes the name of the dataset, then gets the missing values mask, and then fill the data accordingly.
    :param data_name: the dataset name (tabular or graph).
    :return: the data with filled values.
    """

    # case 1: graph dataset, hence we have edges.
    if data_name in GRAPH_DATA_NAMES:
        # get the dataset.
        dataset, _ = get_dataset(data_name)
        data = dataset.data  # get data.
        edges = data.edge_index  # get edges.
        x = pd.DataFrame(data.x)  # get features.

        mask = find_mask(x)  # get mask.

        x = torch.from_numpy(x.values.astype(np.float32))  # convert data to torch tensor.
        x = FeaturePropagation(num_iterations=40).propagate(x, edges, mask)  # fill data.
        return pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(data.y)], axis=1)  # return filled data.

    # case 2: tabular dataset, hence we need to create edges.
    # get DataFrame of the set.
    data_ = get_dataset(data_name)  # get DataFrame of the file data.
    data = data_.copy()
    # z-score.
    data = z_score(data)
    y = data.iloc[:, -1]  # get labels.
    y = torch.from_numpy(y.values.astype(np.int64))  # convert labels to torch tensor.
    x = data.drop(data.columns[-1], axis=1)  # get data.

    # calculate edges.
    edges = make_feature_edges(data)
    # calculate mask.
    mask = find_mask(x)

    x = torch.from_numpy(x.values.astype(np.float32))  # convert data to torch tensor.
    x = FeaturePropagation(num_iterations=40).propagate(x, edges, mask)  # fill data.

    return pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)  # return filled data.


# remove data and fill using FP.
def remove_and_fill(data_name, missing_rate: float = 0.1,):
    """
    this method takes the name of the dataset, then removes the missing values, and then fill the data accordingly.
    :param data_name: name of the dataset.
    :param missing_rate: the rate of missing values.
    :return: the data after filling with FP.
    """

    device = torch.device(
        f"cuda:0"
        if torch.cuda.is_available() else "cpu"
    )

    # if graph dataset.
    if data_name in GRAPH_DATA_NAMES:
        # fetch data and sizes.
        dataset, _ = get_dataset(data_name)
        data = dataset.data
        n_nodes = data.x.shape[0]
        n_features = data.x.shape[1]
        # get a mask.
        mask = get_missing_feature_mask(missing_rate, n_nodes, n_features)
        x = data.x.clone()
        x = pd.DataFrame(x.cpu().detach().numpy())
        # z-score.
        x = z_score(x)
        x = torch.from_numpy(x.values.astype(np.float32)).to(device)

        # remove data.
        x[~mask] = np.nan
        y = data.y

        # convert to DataFrame.
        x = pd.DataFrame(x.cpu().detach().numpy())
        data_unfilled = pd.concat([x, pd.DataFrame(y)], axis=1)

        # filling missing data.
        x = torch.from_numpy(x.values.astype(np.float32)).to(device)
        edges = data.edge_index

        x = FeaturePropagation(num_iterations=40).propagate(x, edges, mask)

        data_filled = pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)

        return data_unfilled, data_filled

    # else it's a tabular dataset.
    data_ = get_dataset(data_name)
    data = data_.copy()

    # shuffle.
    data = shuffle(data)

    # z-score.
    data = z_score(data)

    # remove data.
    data, mask = remove_random_cells(data, missing_rate)

    data_unfilled = data.copy()

    x = data.drop(data.columns[-1], axis=1)

    y = torch.from_numpy(data.iloc[:, -1].values.astype(np.int64)).to(device)

    # filling data.
    edges = make_feature_edges(data)
    x = torch.from_numpy(x.values.astype(np.float32)).to(device)

    model = FeaturePropagation(num_iterations=40)

    x = model.propagate(x, edges, mask)
    # return filled data.
    return data_unfilled, pd.concat([pd.DataFrame(x.cpu().detach().numpy()), pd.DataFrame(y)], axis=1)


def run_xgb(train: pd.DataFrame, test: pd.DataFrame):
    """this method takes as input a DataFrame of train and test sets, initialize a built-in XGB classifier,
    and trains it with the data, the method returns the AUC score of the XGB classifier over the test set."""

    # separate the labels from the data, in both train and test parts. (labels are y_, and data is x_).
    x_train, y_train = (train.iloc[:, :-1].to_numpy().astype(np.float32),
                        train.iloc[:, -1].to_numpy().astype(np.float32))
    x_test = test.iloc[:, :-1].to_numpy().astype(np.float32)
    true_labels = test.iloc[:, -1].to_numpy().astype(np.float32)

    # make predictions, and convert to nearest value in {0,1} in order to compare with the true labels.
    y_test = true_labels

    if len(np.unique(y_test)) == 2:
        # initialize new XGB classifier.
        model = XGBClassifier(n_estimators=30)
        # train the model.
        model.fit(x_train, y_train)

        # make predictions, and convert to nearest value in {0,1} in order to compare with the true labels.
        y_test = true_labels
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)

        test_score = roc_auc_score(y_test, y_pred_test)
        train_score = roc_auc_score(y_train, y_pred_train)

        return train_score, test_score

    # initialize new XGB classifier.
    model = XGBClassifier(n_estimators=30, num_class=len(np.unique(y_train)), objective='multi:softmax',)
    # train the model.
    model.fit(x_train, y_train)

    y_pred_test = model.predict_proba(x_test)
    y_pred_train = model.predict_proba(x_train)
    # gets AUC score.
    test_score = roc_auc_score(y_test, y_pred_test, multi_class='ovr', average="macro")
    train_score = roc_auc_score(y_train, y_pred_train, multi_class='ovr', average="macro")
    return train_score, test_score


# define a wrapper method for run_xgb.
def test_data_(data: pd.DataFrame):
    """
    this method takes a DataFrame, split it to train and test, and uses run_xgb.
    :param data: the dataset (features and labels).
    :return: xgb score.
    """
    train, test = train_test_split(data, test_size=0.2)
    return run_xgb(train, test)[1]  # we only care about the test score.
