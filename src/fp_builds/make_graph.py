import numpy as np
import pandas as pd
import torch


def reg_distance(x: np.ndarray, y: np.ndarray) -> float:
    # returns the Euclidean distance between x and y, where nan is replaced by 0.
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    return np.linalg.norm(x - y)


# novel metric for measuring the distance between two sparse vectors.
def sparse_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate custom distance between two arrays with possible NaN values.

    Parameters:
    - x, y: Input arrays

    Returns:
    - distance: Euclidean distance between x and y
    """

    nan_mask = np.isnan(x) | np.isnan(y)
    distances = np.zeros(x.shape[0])

    nan_mask_2 = np.isnan(x) & ~np.isnan(y)
    nan_mask_3 = ~np.isnan(x) & np.isnan(y)
    num_mask = ~nan_mask

    distances[nan_mask] = 2
    distances[nan_mask_2] = 1 + y[nan_mask_2] ** 2
    distances[nan_mask_3] = 1 + x[nan_mask_3] ** 2
    distances[num_mask] = (x[num_mask] - y[num_mask]) ** 2

    return np.sqrt(np.sum(distances))


def make_graph(frame: pd.DataFrame, k: int = 3):
    """takes a dataframe and returns an adjacency matrix of the edges."""
    frame_ = frame.values
    distances = np.zeros((frame_.shape[0], frame_.shape[0]))

    for i in range(frame_.shape[0]):
        distances[i, :] = [sparse_distance(frame_[i], frame_[j]) for j in range(frame_.shape[0])]

    edges = np.argsort(distances, axis=1)[:, 1:k + 1]  # Changed to get top k distances

    edges_ = np.column_stack((np.repeat(np.arange(edges.shape[0]), k), edges.flatten()))

    edges = torch.tensor(edges_).t().contiguous()

    return edges, distances


def make_graph_random(frame: pd.DataFrame):
    """for each row i, add the edges [i, j], [i,k], [i,l] where j, k, l are random numbers between 0 and the number
    of rows."""
    # for each row, add 4 random edges in the form of [i, j] where i is the row number and j is a random number
    # between 0 and the number of rows.
    edges = []
    for i in range(0, frame.shape[0]):
        for j in range(0, 3):
            x = np.random.randint(0, frame.shape[0])
            edges.append([i, x])
            edges.append([x, i])
    edges = torch.tensor(edges).t().contiguous()
    return edges


def make_graph_forward(frame: pd.DataFrame):
    """for each row i, add the edges [i, i+1], [i, i+2], [i, i+3]. and then make the graph undirected."""
    # for each row, add the following 4 edges: [i, i+1], [i, i+2], [i, i+3]
    edges = []
    for i in range(0, frame.shape[0]):
        edges.append([i, (i + 1) % frame.shape[0]])
        edges.append([i, (i + 2) % frame.shape[0]])
        edges.append([i, (i + 3) % frame.shape[0]])
        edges.append([(i + 1) % frame.shape[0], i])
        edges.append([(i + 2) % frame.shape[0], i])
        edges.append([(i + 3) % frame.shape[0], i])

    edges = torch.tensor(edges).t().contiguous()
    return edges


def make_graph_backward(frame: pd.DataFrame):
    """for each row i, add the edges [i, i-2], [i, i-1], [i, i+1], and then makes the graph undirected."""
    # for each row, add the following 4 edges: [i, i-2], [i, i-1], [i, i+1]
    edges = []
    for i in range(0, frame.shape[0]):
        edges.append([i, (i - 2) % frame.shape[0]])
        edges.append([i, (i - 1) % frame.shape[0]])
        edges.append([i, (i + 1) % frame.shape[0]])
        edges.append([(i - 2) % frame.shape[0], i])
        edges.append([(i - 1) % frame.shape[0], i])
        edges.append([(i + 1) % frame.shape[0], i])

    edges = torch.tensor(edges).t().contiguous()
    return edges


def make_graph_complete(frame: pd.DataFrame):
    """connects every 2 distinct nodes."""
    edges = []
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[0]):
            if i != j:
                edges.append([i, j])
    edges = torch.tensor(edges).t().contiguous()
    return edges


def find_mean_dist(frame: pd.DataFrame, edges: torch.Tensor):
    """Returns a matrix containing the mean distance between each vertex and its neighbors."""
    frame_values = np.nan_to_num(frame.values)
    num_vertices = frame_values.shape[0]
    distances = np.zeros(num_vertices)

    edges_ = edges.t().contiguous()
    # Convert the PyTorch tensor to a NumPy array
    edges_np = edges_.numpy()

    for i in range(num_vertices):
        # Find indices where i is present in either column of edges
        indices_i_in_edges = np.where((edges_np[:, 0] == i) | (edges_np[:, 1] == i))[0]

        # Extract neighbors of vertex i from the edges tensor
        neighbors = np.unique(edges_np[indices_i_in_edges])
        neighbors = neighbors[neighbors != i]  # Exclude the vertex itself

        # Calculate mean distance using vectorized operations
        distances[i] = np.mean(np.linalg.norm(frame_values[i] - frame_values[neighbors], axis=1))

    return distances


def make_graph_knn(frame: pd.DataFrame, k: int = 3, dist: str = "heur"):
    """
    Returns a graph with k-nearest neighbors for each vertex.
    :param dist: distance metric to use for finding nearest neighbors.
    :param frame: the original data, and the data with the existence matrix.
    :param k: the number of neighbors to connect each vertex to.
    :return: a tensor of edges, and a matrix of the distances.
    """

    if dist == "euc":
        data_ = np.nan_to_num(frame.values)
        data = data_.copy()[:, : data_.shape[1] // 2]

        data = pd.DataFrame(data).copy()

        dists = calc_l2(data)

        # Exclude self-distances and get the indices of the top k nearest neighbors
        edges_indices = np.argsort(dists, axis=1)[:, 1:k + 1]

        # Create edges tensor with both [i, j] and [j, i]
        edges_ = np.column_stack((np.repeat(np.arange(edges_indices.shape[0]), k), edges_indices.flatten()))
        edges = np.vstack((edges_, edges_[:, [1, 0]]))  # Include [j, i] for every [i, j]

        # Convert to torch.Tensor
        edges = torch.tensor(edges).t().contiguous()

        return edges, dists

    elif dist == "l_n":
        data = frame.copy().iloc[:, : frame.shape[1] // 2]
        distances = calc_l2_neglecting_nans(pd.DataFrame(data).copy())
        edges = np.argsort(distances, axis=1)[:, 1:k + 1]
        edges_ = np.column_stack((np.repeat(np.arange(edges.shape[0]), k), edges.flatten()))
        edges = np.vstack((edges_, edges_[:, [1, 0]]))  # Include [j, i] for every [i, j]
        edges = torch.tensor(edges).t().contiguous()
        return edges, distances

    data_ = np.nan_to_num(frame.values)
    data_all = data_.copy()

    existence_all = data_all[:, data_all.shape[1] // 2:]

    data_all = data_all[:, : data_all.shape[1] // 2]

    f = lambda r: np.sqrt(np.power(data_all - r, 2).sum(1))  # this returns an array of distances from each
    # row in data_train to r.

    dists = np.apply_along_axis(f, 1, data_all)

    nac = lambda r: (existence_all + r).sum(1)

    nulls_counter = np.apply_along_axis(nac, 1, existence_all)
    nulls_counter = existence_all.shape[1] * 2 - nulls_counter

    dists = np.sqrt(dists ** 2 + nulls_counter)
    m = np.max(dists)
    for i in range(dists.shape[0]):
        dists[i, i] = m
    edges = np.argsort(dists, 1)[:, : k]

    edges_ = []
    for i in range(edges.shape[0]):
        for j in edges[i]:
            edges_.append([i, j])
            if [j, i] not in edges_:
                edges_.append([j, i])
    edges = torch.tensor(edges_).t().contiguous()

    return edges, dists


def make_graph_by_cov(frame: pd.DataFrame, threshold: float = 0.5):
    """
    Returns a list of edges, where each edge is a tuple of two vertices.
    The edges are decided such that [i, j] is an edge if and only if the correlation between frame[i]
    and frame[j] is greater than the specified threshold.

    :param frame: pd.DataFrame, the input data frame.
    :param threshold: float, correlation threshold for creating edges.
    :return: torch.Tensor, list of edges.
    """
    correlations = np.corrcoef(frame.values)
    edges = np.column_stack(np.where(correlations > threshold))

    return torch.tensor(edges).t().contiguous()


def get_indices_below_percentile(distances: np.ndarray, percentile=2):
    """
    Get indices where values in the 2D array are smaller than a given percentile.

    Parameters:
    - distances: np.ndarray, input 2D array of distances
    - percentile: float, optional (default=98), the percentile threshold

    Returns:
    - indices: list, list of indices [i, j] where distances[i, j] is smaller than the specified percentile
    """
    threshold = np.percentile(distances, percentile)

    # Find indices where distances are smaller than the threshold
    indices = np.where(distances < threshold)

    # Combine row and column indices into a list of [i, j]
    indices_list = list(zip(indices[0], indices[1]))

    # convert to tensor.
    indices_list = torch.tensor(indices_list).t().contiguous()

    return indices_list


def calc_l2_neglecting_nans(data_: pd.DataFrame):
    """
    This method calculates the L2 distances between rows in the data when NaN values are neglected.
    :param data_: the data.
    :return: matrix of distances.
    """
    data = data_.copy()
    data = data.fillna(np.nan)  # Fill NaN values to ensure consistent handling
    data = data.values
    num_rows, num_cols = data.shape
    distances = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(num_rows):
            mask_numeric = ~np.isnan(data[i]) & ~np.isnan(data[j])

            if np.sum(mask_numeric) == 0:
                continue

            distances[i, j] = np.linalg.norm(data[i, mask_numeric] - data[j, mask_numeric])
            distances[i, j] *= distances[i, j] * data.shape[1] / np.sum(mask_numeric)
            distances[i, j] = np.sqrt(distances[i, j])
    return distances


def calc_l2(data: pd.DataFrame):
    """
    This method calculates the L2 distances between rows in the data when NaN values are replaced with 0.
    :param data: the data.
    :return: matrix of distances.
    """
    data = data.fillna(0)
    dists = np.linalg.norm(data.values[:, np.newaxis, :] - data.values[np.newaxis, :, :], axis=-1)

    return dists


def calc_heur(data: pd.DataFrame):
    """
    this method calculates heur_dist between rows in the data.
    :param data: the dataset with removed values.
    :return: matrix of distances.
    """
    z = data.notna().astype(int)
    d = calc_l2(data.copy())

    d = np.square(d)
    nac = lambda r: (z + r).sum(1)

    nulls_counter = np.apply_along_axis(nac, 1, z)
    nulls_counter = z.shape[1] * 2 - nulls_counter

    d += nulls_counter
    d = np.sqrt(d)

    return d


def get_knn_edges(distances, k: int = 3):
    """
    this method gets the distances between each row in the data, and returns the tensor of knn edges.
    :param distances: distances between each row in the data.
    :param k: number of neighbors.
    :return: edges of knn graph of the data.
    """
    edges = np.argsort(distances, axis=1)[:, 1:k + 1]
    edges_ = np.column_stack((np.repeat(np.arange(edges.shape[0]), k), edges.flatten()))
    edges = np.vstack((edges_, edges_[:, [1, 0]]))  # Include [j, i] for every [i, j]
    edges = torch.tensor(edges).t().contiguous()
    return edges


def custom_distance_matrix(data, c1, c2):  # todo- check if this method is correct.
    # find nan values
    mask = pd.DataFrame(data.copy().isna().astype(int)).values

    # convert to numpy array and replace nan with 0
    data = data.fillna(0).values

    dists = calc_l2(pd.DataFrame(data)) ** 2

    # calculates penalty for each pair of rows
    c2_penalty = calc_l2(pd.DataFrame(mask)) ** 2  # penalty for pairs with one missing.
    c1_penalty = np.matmul(mask, mask.T)  # penalty for pairs with both missing.

    # add the penalties to the l2 distances
    dists = dists + c1 * c1_penalty + c2 * c2_penalty

    return dists
