from src.temp_tester import *
import warnings

warnings.filterwarnings("ignore")

REMOVED_PERCENTAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
PERCENTAGES = [int(100 * x) for x in REMOVED_PERCENTAGES]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    names = ["Banknote", "Sonar", "Redwine", "Whitewine"]

    for name in names:
        print(name)
        # initialize the result lists.
        unfilled = []
        unfilled_errs = []
        gcnc = []
        gcnc_errs = []
        gnc = []
        gnc_errs = []
        gc = []
        gc_errs = []
        full_score = 0
        for model in ["gc", "gnc", "gc+nc"]:

            a, b, c = test_spagog_results(name, REMOVED_PERCENTAGES, model_name=model, iters=10)
            full_score += a
            if model == "gc":
                gc.append(b)
                gc_errs.append(c)
            elif model == "gnc":
                gnc.append(b)
                gnc_errs.append(c)
            elif model == "gc+nc":
                gcnc.append(b)
                gcnc_errs.append(c)
            else:
                unfilled.append(b)
                unfilled_errs.append(c)
        full_score /= 4
        # print the results.
        print("Done!")
        plt.axhline(y=full_score, color="r", linestyle="--", label="Full data")  # plot the full data score.
        plt.errorbar(REMOVED_PERCENTAGES, unfilled, unfilled_errs, label="Unfilled")  # plot the unfilled data score.
        plt.errorbar(REMOVED_PERCENTAGES, gcnc, gcnc_errs, label="GC+NC")  # plot the gc+nc score.
        plt.errorbar(REMOVED_PERCENTAGES, gnc, gnc_errs, label="GNC")  # plot the gnc score.
        plt.errorbar(REMOVED_PERCENTAGES, gc, gc_errs, label="GC")  # plot the gc score.
        plt.xlabel("Rate of missing values")
        plt.ylabel("AUC score")
        plt.legend()
        plt.grid()
        plt.title(f"Spagog in {name}")
        plt.savefig(f"table_to_graph_plots/Spagog/{name}.png")
        plt.show()

# TODO- fix the issue with datasets.
# TODO- update the readme file.
