import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import optuna
from torch.nn.functional import softmax


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layers):
        super(Net, self).__init__()
        self.fc = []
        self.fc.append(torch.nn.Linear(n_feature, n_hidden))
        self.fc.append(torch.nn.ReLU())
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc.append(torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(n_hidden, n_output)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        for layer in self.fc:
            x = layer(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# initialize a neural network and train it. then optimize the hyperparameters of the neural network using the auc
# score of the validation set, and then evaluate the auc score of the test set.
def run_nn(train: pd.DataFrame, test: pd.DataFrame):
    """takes a training, validation and test set, number of hidden layers, epochs, learning rate and momentum and
    trains a neural network on the training set. the neural network is then optimized using the validation set and
    the auc score is calculated on the test set and returned."""

    # separate the labels from the data, in both train and test parts. (labels are y_, and data is x_).
    train_set, val_set = train_test_split(train, test_size=0.2)

    x_train, y_train = (train_set.iloc[:, :-1].to_numpy().astype(np.float32),
                        train_set.iloc[:, -1].to_numpy().astype(np.float32))
    x_val, y_val = (val_set.iloc[:, :-1].to_numpy().astype(np.float32),
                    val_set.iloc[:, -1].to_numpy().astype(np.float32))

    # Convert NumPy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(test.iloc[:, :-1].to_numpy().astype(np.float32), dtype=torch.float32)
    y_test = test.iloc[:, -1].to_numpy().astype(np.float32)

    # Create DataLoader for train and test datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    def objective(trial):

        n_hidden = trial.suggest_int('n_hidden', 4, 10, 2)
        lr = trial.suggest_uniform('lr', 0.001, 0.1)
        momentum = trial.suggest_uniform('momentum', 0.1, 0.9)
        n_layers = trial.suggest_int('n_layers', 2, 5, 1)

        net = Net(train.shape[1] - 1, n_hidden, len(np.unique(y_train)), n_layers)

        # initialize the optimizer.
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        # initialize the loss function.
        loss_func_ = torch.nn.NLLLoss()

        early_stop_threshold_ = 0.00001
        counter_no_improvement_ = 0
        prev_loss_ = 0
        # train the neural network.
        for epoch_ in range(30):
            for batch_idx_, (data_, target_) in enumerate(train_loader):
                optimizer.zero_grad()
                output_ = net(data_)
                loss_ = loss_func_(output_, target_.long())
                loss_.backward()
                optimizer.step()
            # Check for early stopping.
            if epoch_ > 0 and abs(prev_loss_ - loss_.item()) < early_stop_threshold_:
                counter_no_improvement_ += 1
                if counter_no_improvement_ >= 3:
                    break
            else:
                counter_no_improvement_ = 0
            prev_loss_ = loss_.item()

        # evaluate on validation set.
        with torch.no_grad():
            _, val_prediction = torch.max(net(x_val_tensor), 1)

        val_prediction = val_prediction.cpu().numpy()
        # Calculate AUC score on the validation set.
        auc_val = roc_auc_score(y_val, val_prediction, multi_class='ovr')
        return auc_val

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    best_params = study.best_params

    best_model = Net(train.shape[1] - 1, best_params['n_hidden'], len(np.unique(y_train)), best_params['n_layers'])
    best_optimizer = optim.SGD(best_model.parameters(), lr=best_params['lr'], momentum=best_params['momentum'])

    early_stop_threshold = 0.00001
    counter_no_improvement = 0
    prev_loss = 0
    loss_func = torch.nn.CrossEntropyLoss()

    # train the neural network.
    for epoch in range(30):
        for batch_idx, (data, target) in enumerate(train_loader):
            best_optimizer.zero_grad()
            output = best_model(data)
            loss = loss_func(output, target.long())
            loss.backward()
            best_optimizer.step()
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
            _, test_prediction = torch.max(best_model(x_test_tensor), 1)
            _, train_prediction = torch.max(best_model(x_train_tensor), 1)
        auc_train = roc_auc_score(y_train, train_prediction.cpu().numpy())
        auc_test = roc_auc_score(y_test, test_prediction.cpu().numpy())
        return auc_train, auc_test
    # multiclass.
    with torch.no_grad():
        test_prediction = softmax(best_model(x_test_tensor), dim=1)
        train_prediction = softmax(best_model(x_train_tensor), dim=1)

    auc_test = roc_auc_score(y_test, test_prediction, multi_class='ovr', average="macro")
    auc_train = roc_auc_score(y_train, train_prediction, multi_class='ovr', average="macro")
    return auc_train, auc_test


# initialize XGBoost and train it with the given training set. then optimize the hyperparameters of the XGB using the
# auc score of the validation set via grid search, and then evaluate it on the test set.
def run_xgb(train: pd.DataFrame, test: pd.DataFrame):
    """this method takes as input a DataFrame of train, validation and test sets, initialize a built-in XGB classifier,
    and trains it with the data, the method returns the AUC score of the XGB classifier over the test set."""

    # separate the labels from the data, in both train and test parts. (labels are y_, and data is x_).
    x_train, y_train = (train.iloc[:, :-1].to_numpy().astype(np.float32),
                        train.iloc[:, -1].to_numpy().astype(np.float32))
    x_test = test.iloc[:, :-1].to_numpy().astype(np.float32)
    true_labels = test.iloc[:, -1].to_numpy().astype(np.float32)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100, 10)
        learning_rate = trial.suggest_uniform('learning_rate', 0.01, 0.1)
        gamma = trial.suggest_uniform('gamma', 0.0, 0.5)
        max_depth = trial.suggest_int('max_depth', 3, 9, 2)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 5, 2)
        max_delta_step = trial.suggest_int('max_delta_step', 0, 5, 1)
        lambda_ = trial.suggest_uniform('lambda', 0.0, 1.0)
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        refresh_leaf = trial.suggest_categorical('refresh_leaf', [0, 1])

        # initialize new XGB classifier.
        model_ = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, gamma=gamma,
                               objective='binary:logistic', eval_metric='auc', max_depth=max_depth,
                               min_child_weight=min_child_weight, reg_lambda=lambda_, alpha=alpha,
                               refresh_leaf=refresh_leaf, max_delta_step=max_delta_step)

        model_.fit(x_train, y_train)

        # evaluate on validation set.

        val_prediction = model_.predict(x_test)
        # Calculate AUC score on the validation set.
        auc_val = roc_auc_score(true_labels, val_prediction)

        return auc_val

    y_test = true_labels

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    best_params = study.best_params
    model = XGBClassifier(n_estimators=best_params['n_estimators'],
                          learning_rate=best_params['learning_rate'], gamma=best_params['gamma'],
                          objective='binary:logistic', eval_metric='auc', max_depth=best_params['max_depth'],
                          min_child_weight=best_params['min_child_weight'], reg_lambda=best_params['lambda'],
                          alpha=best_params['alpha'], refresh_leaf=best_params['refresh_leaf'],
                          max_delta_step=best_params['max_delta_step'])

    # train the model.
    model.fit(x_train, y_train)

    y_pred_test = model.predict_proba(x_test)
    y_pred_train = model.predict_proba(x_train)
    # gets AUC score.
    test_score = roc_auc_score(y_test, y_pred_test, multi_class='ovr', average="macro")
    train_score = roc_auc_score(y_train, y_pred_train, multi_class='ovr', average="macro")
    return train_score, test_score
