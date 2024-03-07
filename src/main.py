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
        # initialize the result lists.
        a1, um, us = test_spagog_results(name, REMOVED_PERCENTAGES, model_name="unfilled", iters=10)
        a4, gncm, gncs = test_spagog_results(name, REMOVED_PERCENTAGES, model_name="gnc", iters=10)
        a3, gcm, gcs = test_spagog_results(name, REMOVED_PERCENTAGES, model_name="gc", iters=10)
        a2, gcncm, gcncs = test_spagog_results(name, REMOVED_PERCENTAGES, model_name="gc+nc", iters=10)
        full_score = sum([a1, a2, a3, a4]) / 4
        # print the results.
        print("Done!")
        plt.axhline(y=full_score, color="r", linestyle="--", label="Full data")  # plot the full data score.
        plt.errorbar(REMOVED_PERCENTAGES, um, us, label="Unfilled")  # plot the unfilled data score.
        plt.errorbar(REMOVED_PERCENTAGES, gcncm, gcncs, label="GC+NC")  # plot the gc+nc score.
        plt.errorbar(REMOVED_PERCENTAGES, gncm, gncs, label="GNC")  # plot the gnc score.
        plt.errorbar(REMOVED_PERCENTAGES, gcm, gcs, label="GC")  # plot the gc score.
        plt.xlabel("Rate of missing values")
        plt.ylabel("AUC score")
        plt.legend()
        plt.grid()
        plt.title(f"Spagog in {name}")
        plt.savefig(f"table_to_graph_plots/Spagog/{name}.png")
        plt.show()

# TODO- fix the issue with datasets.
# TODO- update the readme file. (not urgent)
