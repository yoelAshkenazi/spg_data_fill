import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.temp_tester import *
import warnings

warnings.filterwarnings("ignore")

REMOVED_PERCENTAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
PERCENTAGES = [int(100 * x) for x in REMOVED_PERCENTAGES]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    names = ["Redwine", "Wine", "Accent"]
    # for name in names:
    #     draw_final_results(name, [0.1], REMOVED_PERCENTAGES, 10)


# TODO- fix the issue with datasets.
