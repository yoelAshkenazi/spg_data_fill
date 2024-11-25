import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.data import Data


# initialize a neural network with 3 hidden layers, and evaluate it.
# the neural network is trained on 80% of the data, and tested on the remaining 20%.

class Net(nn.Module):
    def __init__(self, n_features, n_output):
        super(Net, self).__init__()
        hidden = n_features // 2
        self.fc1 = (torch.nn.Linear(n_features, hidden))
        self.activation1 = (torch.nn.Tanh())
        if n_output == 2:
            self.final = (torch.nn.Linear(hidden, 1))
            self.out = torch.nn.Sigmoid()
        else:
            self.final = (torch.nn.Linear(hidden, n_output))
            self.out = torch.nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.final(x)
        x = self.out(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_output):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_features, n_features // 2)
        self.conv2 = GCNConv(n_features // 2, n_output)

    def forward(self, _data):
        x, edge_index = _data.x, _data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


def run_nn(train: pd.DataFrame, test: pd.DataFrame):
    """takes a training and test set, number of hidden layers, epochs, learning rate and momentum and trains a neural
    network on the training set. the neural network is then evaluated on the test set and the accuracy is returned."""
    # Extract features and labels from the train and test datasets
    x_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
    x_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values

    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    if torch.cuda.is_available():
        x_train_tensor = x_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        x_test_tensor = x_test_tensor.cuda()

    # Create DataLoader for train and test datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    early_stop_threshold = 0.00001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the neural network.
    net = Net(n_features=x_train.shape[1], n_output=len(np.unique(y_train))).to(device)
    if len(np.unique(y_train)) == 2:
        loss_func = torch.nn.BCELoss()
    else:
        sizes = [len(y_train[y_train == i]) for i in np.unique(y_train)]
        weights = [size / len(y_train) for size in sizes]
        weight = torch.tensor(weights).to(device)
        loss_func = torch.nn.CrossEntropyLoss(weight=weight)
    # initialize the optimizer.
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    counter_no_improvement = 0
    prev_loss = 0
    # train the neural network.
    for epoch in range(50):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = net(data) if len(np.unique(y_train)) > 2 else net(data).squeeze()
            loss = loss_func(output, target.long()) if len(np.unique(y_train)) > 2 else loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Check for early stopping.
        if epoch > 0 and abs(prev_loss - loss.item()) < early_stop_threshold:
            counter_no_improvement += 1
            if counter_no_improvement >= 5:
                break
        else:
            counter_no_improvement = 0
        prev_loss = loss.item()

    # Evaluate on the train set and test set.
    if len(np.unique(y_test)) == 2:
        with torch.no_grad():
            test_prediction = net(x_test_tensor)
            train_prediction = net(x_train_tensor)
        auc_train = roc_auc_score(y_train, train_prediction.cpu().numpy())
        auc_test = roc_auc_score(y_test, test_prediction.cpu().numpy())

        return auc_train, auc_test
    # multiclass.

    with torch.no_grad():
        test_preds = torch.argmax((net(x_test_tensor)), dim=1)
        train_preds = torch.argmax((net(x_train_tensor)), dim=1)
    # evaluate accuracy.
    # auc_train = np.sum(train_preds.cpu().numpy() == y_train) / len(y_train)
    # auc_test = np.sum(test_preds.cpu().numpy() == y_test) / len(y_test)
    # evaluate f1 micro score.
    train_f1 = f1_score(y_train, train_preds.cpu().numpy(), average='weighted')
    test_f1 = f1_score(y_test, test_preds.cpu().numpy(), average='weighted')
    auc_train = train_f1
    auc_test = test_f1
    return auc_train, auc_test


def filter_edge_index(edge_index, node_indices):
    """ Filters edge_index to only include edges where both nodes are in node_indices. """
    mask = torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices)
    return edge_index[:, mask]


def run_gcn(data):
    """takes a training and test set, number of hidden layers, epochs, learning rate and momentum and trains a neural
    network on the training set. the neural network is then evaluated on the test set and the accuracy is returned."""
    # Extract features and labels from the train and test datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)
    model = GCN(n_features=data.num_features, n_output=len(np.unique(data.y))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, preds = model(data).max(dim=1)
    preds_train = preds[data.train_mask]
    preds_test = preds[data.test_mask]
    if len(np.unique(data.y)) == 2:
        auc_train = roc_auc_score(data.y[data.train_mask].cpu().numpy(), preds_train.cpu().numpy())
        auc_test = roc_auc_score(data.y[data.test_mask].cpu().numpy(), preds_test.cpu().numpy())

    else:
        auc_train = f1_score(data.y[data.train_mask].cpu().numpy(), preds_train.cpu().numpy(), average='weighted')
        auc_test = f1_score(data.y[data.test_mask].cpu().numpy(), preds_test.cpu().numpy(), average='weighted')

    return auc_train, auc_test


# initialize XGBoost and train it with the given data, and evaluate it.
# the algorithm is trained on 80% of the data, and tested on the remaining 20%.
def run_xgb(train: pd.DataFrame, test: pd.DataFrame):
    """this method takes as input a DataFrame of train and test sets, initialize a built-in XGB classifier,
    and trains it with the data, the method returns the AUC score of the XGB classifier over the test set."""

    # separate the labels from the data, in both train and test parts. (labels are y_, and data is x_).
    x_train, y_train = (train.iloc[:, :-1].to_numpy().astype(np.float32),
                        train.iloc[:, -1].to_numpy().astype(np.float32))
    x_test = test.iloc[:, :-1].to_numpy().astype(np.float32)
    y_test = test.iloc[:, -1].to_numpy().astype(np.float32)

    if len(np.unique(y_test)) == 2:
        # initialize new XGB classifier.
        model = XGBClassifier(n_estimators=10)
        # train the model.
        model.fit(x_train, y_train)

        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)

        test_score = roc_auc_score(y_test, y_pred_test)
        train_score = roc_auc_score(y_train, y_pred_train)

        return train_score, test_score

    # initialize new XGB classifier.
    model = XGBClassifier(n_estimators=10, num_class=len(np.unique(y_train)), objective='multi:softmax',)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights
    # train the model.
    model.fit(x_train, y_train,)

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    # gets accuracy score.
    # test_score = np.sum(y_pred_test == y_test) / len(y_test)
    # train_score = np.sum(y_pred_train == y_train) / len(y_train)
    test_score = f1_score(y_test, y_pred_test, average='weighted')
    train_score = f1_score(y_train, y_pred_train, average='weighted')

    return train_score, test_score


def run_with_preds_nn(train: pd.DataFrame, test: pd.DataFrame):
    """this method takes as input a train and test set, and a function, the function is trained on the train set,
    and evaluated on the test set, the method returns the AUC score of the function over the test set."""
    # Extract features and labels from the train and test datasets
    x_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
    x_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values
    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    if torch.cuda.is_available():
        x_train_tensor = x_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        x_test_tensor = x_test_tensor.cuda()
    # Create DataLoader for train and test datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    early_stop_threshold = 0.00001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize the neural network.
    net = Net(n_features=x_train.shape[1], n_output=len(np.unique(y_train))).to(device)
    loss_func = torch.nn.CrossEntropyLoss() if len(np.unique(y_train)) > 2 else torch.nn.BCELoss()
    # initialize the optimizer.
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    counter_no_improvement = 0
    prev_loss = 0
    # train the neural network.
    for epoch in range(50):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = net(data) if len(np.unique(y_train)) > 2 else net(data).squeeze()
            loss = loss_func(output, target.long()) if len(np.unique(y_train)) > 2 else loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Check for early stopping.
        if epoch > 0 and abs(prev_loss - loss.item()) < early_stop_threshold:
            counter_no_improvement += 1
            if counter_no_improvement >= 5:
                break
        else:
            counter_no_improvement = 0
        prev_loss = loss.item()
    # Evaluate on the train set and test set.
    if len(np.unique(y_test)) == 2:
        with torch.no_grad():
            test_prediction = net(x_test_tensor)
            train_prediction = net(x_train_tensor)
        auc_train = roc_auc_score(y_train, train_prediction.cpu().numpy())
        auc_test = roc_auc_score(y_test, test_prediction.cpu().numpy())
        return auc_train, auc_test, test_prediction
    # multiclass.
    # evaluate accuracy.
    with torch.no_grad():
        test_preds = torch.argmax((net(x_test_tensor)), dim=1)
        train_preds = torch.argmax((net(x_train_tensor)), dim=1)
    auc_train = np.sum(train_preds.cpu().numpy() == y_train) / len(y_train)
    auc_test = np.sum(test_preds.cpu().numpy() == y_test) / len(y_test)
    test_preds = test_preds.cpu().numpy()
    return auc_train, auc_test, test_preds


def run_with_preds_xgb(train: pd.DataFrame, test: pd.DataFrame):
    """this method takes as input a DataFrame of train and test sets, initialize a built-in XGB classifier,
    and trains it with the data, the method returns the AUC score of the XGB classifier over the test set."""
    # separate the labels from the data, in both train and test parts. (labels are y_, and data is x_).
    x_train, y_train = (train.iloc[:, :-1].to_numpy().astype(np.float32),
                        train.iloc[:, -1].to_numpy().astype(np.float32))
    x_test = test.iloc[:, :-1].to_numpy().astype(np.float32)
    y_test = test.iloc[:, -1].to_numpy().astype(np.float32)
    if len(np.unique(y_test)) == 2:
        # initialize new XGB classifier.
        model = XGBClassifier(n_estimators=10)
        # train the model.
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        test_score = roc_auc_score(y_test, y_pred_test)
        train_score = roc_auc_score(y_train, y_pred_train)
        return train_score, test_score, y_pred_test
    # initialize new XGB classifier.
    model = XGBClassifier(n_estimators=10, num_class=len(np.unique(y_train)), objective='multi:softmax',)
    # train the model.
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    # gets accuracy score.
    test_score = np.sum(y_pred_test == y_test) / len(y_test)
    train_score = np.sum(y_pred_train == y_train) / len(y_train)
    return train_score, test_score, y_pred_test
