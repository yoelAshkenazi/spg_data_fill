from src.temp_tester import *
import warnings
from scipy.io import arff
warnings.filterwarnings("ignore")

REMOVED_PERCENTAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
PERCENTAGES = [int(100 * x) for x in REMOVED_PERCENTAGES]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    name = 'Segment'
    # draw_final_results("earthquakes", REMOVED_PERCENTAGES, PERCENTAGES, 10)
    for name in os.listdir('data/binary'):

        print(name)
        name = name.split('.')[0]
        # fix the labels to start from 0
        df = pd.read_csv(f'data/binary/{name}.csv')
        df.iloc[:, -1] = df.iloc[:, -1] - min(df.iloc[:, -1])  # fix the labels to start from 0
        df.to_csv(f'data/binary/{name}.csv', index=False)
        # test_filling_auc(name, REMOVED_PERCENTAGES, 10, False)
        test_filling_auc_gcn(name, REMOVED_PERCENTAGES, 10, False)
        # compare_xgb_gcn(name)
        print('-' * 50)
    exit()
    # make_lineplots()
    make_lineplots_gcn()

    # import json
    # results_list = []
    # for file in os.listdir('auc results'):
    #     with open(f'auc results/{file}', 'rb') as f:
    #         results_list.append(json.load(f))
    #
    # plot_auc_per_percentage(results_list)

    # # Load the .arff file
    # data, meta = arff.loadarff(f'data/multiclass/{name}.arff')
    #
    # df = pd.DataFrame(data)
    #
    # # Save to CSV
    # df.to_csv(f'data/multiclass/{name}.csv', index=False)
