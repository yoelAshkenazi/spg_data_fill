import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax


# initialize a neural network with 3 hidden layers, and evaluate it.
# the neural network is trained on 80% of the data, and tested on the remaining 20%.

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
    net = torch.nn.Sequential(
        torch.nn.Linear(train.shape[1] - 1, train.shape[1] - 1),
        torch.nn.ReLU(),
        torch.nn.Linear(train.shape[1] - 1, train.shape[1] // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(train.shape[1] // 2, len(np.unique(y_train))),
        torch.nn.LogSoftmax(dim=1)
    ).to(device)
    # initialize the optimizer.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # initialize the loss function.
    loss_func = torch.nn.NLLLoss()

    counter_no_improvement = 0
    prev_loss = 0
    # train the neural network.
    for epoch in range(30):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = loss_func(output, target.long())
            loss.backward()
            optimizer.step()
        # Check for early stopping.
        if epoch > 0 and abs(prev_loss - loss.item()) < early_stop_threshold:
            counter_no_improvement += 1
            if counter_no_improvement >= 3:
                break
        else:
            counter_no_improvement = 0
        prev_loss = loss.item()

    # Evaluate on the train set and test set.
    if len(np.unique(y_test)) == 2:
        with torch.no_grad():
            _, test_prediction = torch.max(net(x_test_tensor), 1)
            _, train_prediction = torch.max(net(x_train_tensor), 1)
        auc_train = roc_auc_score(y_train, train_prediction.cpu().numpy())
        auc_test = roc_auc_score(y_test, test_prediction.cpu().numpy())

        return auc_train, auc_test
    # multiclass.
    with torch.no_grad():
        test_prediction = softmax(net(x_test_tensor), dim=1)

        train_prediction = softmax(net(x_train_tensor), dim=1)

    auc_test = roc_auc_score(y_test, test_prediction, multi_class='ovr', average="macro")
    auc_train = roc_auc_score(y_train, train_prediction, multi_class='ovr', average="macro")
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
    true_labels = test.iloc[:, -1].to_numpy().astype(np.float32)

    # make predictions, and convert to nearest value in {0,1} in order to compare with the true labels.
    y_test = true_labels

    if len(np.unique(y_test)) == 2:
        # initialize new XGB classifier.
        model = XGBClassifier(n_estimators=10)
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
    model = XGBClassifier(n_estimators=10, num_class=len(np.unique(y_train)), objective='multi:softmax',)
    # train the model.
    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    # gets accuracy score.
    test_score = np.sum(y_pred_test == y_test) / len(y_test)
    train_score = np.sum(y_pred_train == y_train) / len(y_train)
    return train_score, test_score
