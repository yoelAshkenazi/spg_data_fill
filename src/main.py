from src.temp_tester import *
import pandas as pd
import build_classifier as classify
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

CITE_FP_SCORES = [0.6732, 0.6677, 0.6781, 0.671, 0.6735, 0.6555, 0.6503, 0.6532, 0.6535]
PUBMED_FP_SCORES = [0.7589, 0.7548, 0.7563, 0.7472, 0.7491, 0.7511, 0.7513, 0.749, 0.7447]
CORA_FP_SCORES = [0.8089, 0.8008, 0.8053, 0.7961, 0.7998, 0.7992, 0.7947, 0.7925, 0.7829]

REMOVED_PERCENTAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    directory = os.path.dirname(__file__) + "/Datasets/data.csv"

    # todo- fix FP. problem : issue with transforming the data to a graph.
    # note- make it faster(?)
    # frame = dp.csv_to_df(directory)

    # run the classifier on the training and test set.
    # a1, a1_ = plot.auc(frame.copy(), REMOVED_PERCENTAGES, classify.run_nn, "naive NN")
    # a2, a2_ = plot.auc(frame.copy(), REMOVED_PERCENTAGES, classify.run_xgb, "naive XGB")
    # b1, b1_ = plot.auc(frame.copy(), REMOVED_PERCENTAGES, classify_opt.run_nn, "optimized NN")
    # b2, b2_ = plot.auc(frame.copy(), REMOVED_PERCENTAGES, classify_opt.run_xgb, "optimized XGB")
    # c1, c1_ = plot.auc_imputation(frame.copy(), REMOVED_PERCENTAGES, classify.run_nn, "NN imputed")
    # c2, c2_ = plot.auc_imputation(frame.copy(), REMOVED_PERCENTAGES, classify.run_xgb, "XGB imputed")
    # d1, d1_ = plot.auc_imputation(frame.copy(), REMOVED_PERCENTAGES, classify_opt.run_nn,
    #                               "optimized NN imputed")
    # d2, d2_ = plot.auc_imputation(frame.copy(), REMOVED_PERCENTAGES, classify_opt.run_xgb,
    #                               "optimized XGB imputed")
    # define labels.
    labels_one_method = ["train", "test"]
    labels_xgb = ["naive XGB", "optimized XGB"]
    labels_nn = ["naive NN", "optimized NN"]
    labels_imp_ = ["regular", "imputed", ]
    labels = ["naive NN", "naive XGB", "optimized NN", "optimized XGB"]
    labels_ = ["naive NN imputed", "naive XGB imputed", "optimized NN imputed", "optimized XGB imputed"]
    labels_imp = ["naive NN", "naive XGB", "naive NN imputed", "naive XGB imputed"]
    temp_labels = ["naive xgb", "FP model", "naive xgb with fp features"]

    # run the plotting method.
    PERCENTAGES = [int(100 * x) for x in REMOVED_PERCENTAGES]

    # results.

    # fix labels in all data so the labels range from 0 to the number of classes.
    # first get a list of every file in the directory.

    names = ["Banknote", "sonar", "RedWine", "WhiteWine"]

    for name in names:
        print(name)
        get_best_constants(name, REMOVED_PERCENTAGES)

    # get_best_constants(name, REMOVED_PERCENTAGES)

# TODO- fix the issue with datasets.
# TODO- update the readme file.
