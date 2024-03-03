from src.temp_tester import *
import warnings

warnings.filterwarnings("ignore")

REMOVED_PERCENTAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
PERCENTAGES = [int(100 * x) for x in REMOVED_PERCENTAGES]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    names = ["Redwine", "Whitewine"]

    for name in names:
        print(name)
        get_best_constants(name, REMOVED_PERCENTAGES)

# TODO- fix the issue with datasets.
# TODO- update the readme file.
