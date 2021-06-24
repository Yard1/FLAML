'''Requirements if run exp that use deeptable model: 
conda create -n py37 python=3.7
pip install flaml[blendsearch,ray]
pip install tensorflow==2.2.0 deeptables[gpu]
pip install openml==0.9
pip install pandas==1.0
# pip install CondfigSpace hpbandster
# Find /lib/python3.7/site-packages/deeptables/preprocessing/transformer.py
#     # comment line 40:
#     # y = y.argmax(axis=-1)
'''

import time
import numpy as np
from flaml import tune
import pandas as pd
from functools import partial
import os
import logging
RANDOMSEED = 1024
try:
    import ray
    import flaml
    from flaml import BlendSearch, BlendSearchNoDiscount
except ImportError:
    print("pip install flaml[blendsearch,ray,openml]")
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import os
import json


def get_lc_from_log(log_file_name, budget=np.inf):
    x = []
    y = [] #minimizing 
    if os.path.isfile(log_file_name):
        best_obj = np.inf 
        total_time = 0
        with open(log_file_name) as f:
            for line in f:
                record = json.loads(line)
                if float(record['total_search_time']) > budget: break
                total_time = float(record['total_search_time'])
                obj = float(record['obj'])
                if obj < best_obj: 
                    best_obj = obj
                    x.append(total_time)
                    y.append(best_obj)
            x.append(total_time)
            y.append(best_obj)
    else:
        print('log_file_name', log_file_name)
        print('File does not exist')
    assert len(x) == len(y) 
    return x, y
    

# discretize curve_dic
def discretize_learning_curves(self, time_budget, curve_dic):
    # curve_dic: key: method name, value: list of multiple run
    # each run is a list of tuples run_0 = [ (time, loss) ]

    self.discretized_curve = {}
    for method, curve_folds in curve_dic.items():
        discretized_curve_folds = []
        for curve in curve_folds: # curve in each fold
            # discretize the time domain into intervals of 1s
            discretized_curve = []
            pos = 0
            # discretized_obj = np.iinfo(int).max
            for discretized_t in range(int(time_budget) + 1): # 3600 or 14400
                discretized_obj = np.iinfo(int).max
                while pos < len(curve) and curve[pos][1] <= discretized_t:
                    obj, t, i = curve[pos]
                    discretized_obj = obj
                    pos += 1
                if discretized_obj != np.iinfo(int).max:
                    discretized_curve.append((discretized_obj, discretized_t, -1))
            if len(discretized_curve) == 0: discretized_curve.append((discretized_obj, discretized_t, -1))
            discretized_curve_folds.append(discretized_curve)   
        self.discretized_curve[method] = discretized_curve_folds
    #print (self.discretized_curve)

    
def plot_lc(log_file_name, y_min=0, y_max=0.5, name=''):
    x_list, y_list = get_lc_from_log(log_file_name)
    plt.step(x_list, y_list, where='post', label=name)
    # plt.ylim([y_min,y_max])
    plt.yscale('log')

    print('plot lc')


def get_agg_lc_from_file(log_file_name_alias, method_alias, index_list=list(range(0,10))):
    log_file_list = []
    list_x_list = []
    list_y_list = []
    for index in index_list:
        log_file_name = log_file_name_alias + '_' + str(index) + '.log'
        try:
            x_list, y_list = get_lc_from_log(log_file_name)
            list_x_list.append(x_list)
            list_y_list.append(y_list)
        except:
            print('Fail to get lc from log')
    
    plot_agg_lc(list_x_list, list_y_list, method_alias= method_alias, y_max=1,)


def plot_agg_lc(list_x_list, list_y_list, y_max, method_alias=''):

    def get_x_y_from_index(current_index, x_list, y_list, y_max):
        if current_index ==0:
            current_index_x, current_index_y =  x_list[current_index], y_max
        else: 
            current_index_x, current_index_y = x_list[current_index], y_list[current_index-1]
        return current_index_x, current_index_y

    import itertools
    all_x_list = list(itertools.chain.from_iterable(list_x_list))
    all_x_list.sort()
    all_y_list = []
    for fold_index, y_list in enumerate(list_y_list):
        if len(y_list) == 0:
            continue
        all_y_list_fold = []
        x_list = list_x_list[fold_index]
        current_index = 0
        
        for x in all_x_list:
            current_index_x, best_y_before_current_x = get_x_y_from_index(current_index, x_list, y_list, y_max)
            if x < current_index_x:
                all_y_list_fold.append(best_y_before_current_x)
            else: 
                all_y_list_fold.append(y_list[current_index])
                if current_index < len(y_list)-1:
                    current_index +=1
        all_y_list.append(all_y_list_fold)
    all_y_list_arr = np.array(all_y_list)
    all_y_list_mean = np.mean(all_y_list_arr, axis=0)
    all_y_list_std = np.std(all_y_list_arr, axis=0)

    # plt.plot(all_x_list, all_y_list_mean, label = method_alias)
    plt.step(all_x_list, all_y_list_mean, where='post', label = method_alias)
    plt.fill_between(all_x_list, all_y_list_mean - all_y_list_std, all_y_list_mean + all_y_list_std, alpha=0.4)
    plt.yscale('log')


def log_file_name(problem, dataset, method, budget, run):
    '''log file name
    '''
    return f'logs/{problem}/{problem}_1_1_{dataset}_{budget}_{method}_{run}.log'


def agg_final_result(problems, datasets, methods, budget, run, time=np.inf):
    '''aggregate the final results for problems * datasets * methods
    
    Returns:
        A dataframe with aggregated results
        e.g., problems = ["xgb_blendsearch", "xgb_cfo", "xgb_hpolib"]
        output = df
index | xgb_blendsearch | xgb_cfo         | xgb_hpolib         | winner          | winner_method
blood | (0.2, 1900, BS) | (0.3, 109, CFO) | (0.4, 800, Optuna) | xgb_blendsearch | CFO
        ...
        the numbers correspond to the best loss and minimal time used 
            by all the methods for
            blood * xgb_blendsearch, blood * xgb_cfo, blood * xgb_hpolib etc.
    '''
    results = {}
    columns = problems
    for i, problem in enumerate(problems):
        for j, dataset in enumerate(datasets):
            for method in methods:
                key = (j, i)
                x, y = get_lc_from_log(log_file_name(
                    problem, dataset, method, budget, run), time)
                if len(y)>0: 
                    result = results.get(key, [])
                    if not result: results[key] = result                    
                    result.append((y[-1],x[y.index(y[-1])],method))
    import pandas as pd
    agg_results = pd.DataFrame(columns=columns, index=datasets, dtype=object)
    for key, value in results.items():
        row_id, col_id = key
        agg_results.iloc[row_id, col_id] = min(value)
    best_obj = agg_results.min(axis=1)
    winner = []
    winner_method = []
    for dataset in datasets:
        for problem in problems:
            if agg_results[problem][dataset] == best_obj[dataset]:
                    winner.append(problem)
                    winner_method.append(agg_results[problem][dataset][2])
                    break
    agg_results['winner'] = winner
    agg_results['winner_method'] = winner_method
    return agg_results


def final_result(problem, datasets, methods, budget, run, time=np.inf):
    '''compare the final results for problem on each dataset across methods
    
    Returns:
        A dataframe for each dataset
        e.g., methods = ["CFO", "BlendSearch+Optuna", "Optuna"]
        output = df
            index | CFO | BlendSearch+Optuna | Optuna      | winner
            blood | (0.2, 312) | (0.2, 350)  | (0.4, 1800) | CFO
            ...
        the numbers correspond to the best loss and nimial time used 
            among all the methods for each dataset
    '''
    results = {}
    columns = methods
    import pandas as pd
    agg_results = pd.DataFrame(columns=columns, index=datasets)
    for row_id, dataset in enumerate(datasets):
        for col_id, method in enumerate(methods):
            x, y = get_lc_from_log(log_file_name(
                problem, dataset, method, budget, run), time)
            if len(y)>0: 
                agg_results.iloc[row_id, col_id] = (y[-1],x[y.index(y[-1])])
    best_obj = agg_results.min(axis=1)
    winner = []
    for dataset in datasets:
        for method in methods:
            if agg_results[method][dataset] == best_obj[dataset]:
                    winner.append(f'{method}')
                    break
    agg_results['winner'] = winner
    return agg_results


def test_agg_final_result():
    # print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large', 'lgbm_mlnet'],
    #     ['Australian', 'blood', 'kr'], 
    #     ['BlendSearch+Optuna',],
    #     3600.0, 0))
    # print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large'],
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
    #      'riccardo', 'higgs', 'fabert', 'cnae'], 
    #     ['BlendSearch+Optuna',],
    #     14400.0, 0, 3600))
    # print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large'],
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
    #      'riccardo', 'higgs', 'fabert', 'cnae'], 
    #     ['BlendSearch+Optuna',],
    #     14400.0, 0))
    # print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large', 'lgbm_mlnet_alter'],
    #     ['sylvine', 'mfeat', 'jungle', 
    #      'riccardo', 'fabert', 'cnae'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))

    # print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large', 'lgbm_mlnet'],
    #     ['Australian', 'blood', 'kr'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     3600.0, 0))
    print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large', 'lgbm_large', 'lgbm_mlnet'],
        ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
         'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
         'riccardo', 'higgs', 'fabert', 'cnae'], 
        ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
        14400.0, 0, 3600))
    print(agg_final_result(['lgbm_cfo', 'lgbm_cfo_large', 'lgbm_large', 'lgbm_mlnet'],
        ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
         'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
         'riccardo', 'higgs', 'fabert', 'cnae'], 
        ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
        14400.0, 0))
    # print(agg_final_result(
    #     ['xgb_blendsearch_large', 'xgb_cfo_large'],# 'xgb_blendsearch', 'xgb_cfo', 'xgb_hpolib'],
    #     ['christine', 'jasmine', 'KDDCup09', 'numerai28'], 
    #     # ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine',], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))
    # print(agg_final_result(['xgb_blendsearch', 'xgb_cfo', 'xgb_hpolib'],
    #     ['Australian', 'blood', 'car', 'credit', 'kc1', 'kr', 'phoneme', 'segment'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     3600.0, 0))
    # print(agg_final_result(['xgb_blendsearch', 'xgb_cfo', 'xgb_hpolib'],
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine',], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))
    # print(agg_final_result(['xgb_blendsearch', 'xgb_cfo', 'xgb_hpolib'],
    #     ['guillermo', 'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle',
    #     'jasmine', 'riccardo', 'higgs', 'fabert', 'cnae', ], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))


def test_final_result():
    # print("lgbm_cfo_large")
    # print(final_result('lgbm_cfo_large',
    #     ['Australian', 'blood', 'kr'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     3600.0, 0))
    # print(final_result('lgbm_cfo_large',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))
    print("lgbm, 1h")
    print(final_result('lgbm',
        ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
         'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
         'riccardo', 'higgs', 'fabert', 'cnae',
         'KDDCup09', 'numerai28', 'credit', 'car', 'kc1', 'phoneme', 'segment'], 
        ['BlendSearch+Optuna', 'CFO', 'Optuna'],
        28800.0, 0, 3600))
    print("lgbm, 8h")
    print(final_result('lgbm',
        ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
         'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
         'riccardo', 'higgs', 'fabert', 'cnae',
         'KDDCup09', 'numerai28', 'credit', 'car', 'kc1', 'phoneme', 'segment'], 
        ['BlendSearch+Optuna', 'CFO', 'Optuna'],
        28800.0, 0))
    # print(final_result('lgbm',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine',
    #      'riccardo', 'higgs', 'fabert', 'cnae'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))        
    # print("lgbm_cfo, 1h")
    # print(final_result('lgbm_cfo',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0, 3600))
    # print(final_result('lgbm_cfo',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine'], 
    #     ['BlendSearch+Optuna', 'CFO', 'Optuna'],
    #     14400.0, 0, 3600))
    # print("lgbm_cfo, 10m")
    # print(final_result('lgbm_cfo',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0, 600))
    # print(final_result('lgbm_cfo',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine', 'guillermo',
    #      'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine'], 
    #     ['BlendSearch+Optuna', 'CFO',],
    #     14400.0, 0, 600))
    # print('xgb_cfo')
    # print(final_result('xgb_cfo',
    #     ['Australian', 'blood', 'car', 'credit', 'kc1', 'kr', 'phoneme', 'segment'], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     3600.0, 0))
    # print(final_result('xgb_cfo',
    #     ['Airlines', 'christine', 'shuttle', 'connect', 'sylvine',], 
    #     ['Ax', 'BlendSearch+Optuna', 'CFO', 'HyperOpt', 'Nevergrad', 'Optuna'],
    #     14400.0, 0))


oml_tasks = {
        'adult': {
            'task_id': 7592,
            'task_type': 'binary',
        },
        'Airlines': {
            'task_id': 189354,
            'task_type': 'binary',
        },
        'Amazon': {
            'task_id': 34539,
            'task_type': 'binary',
        },
        'APSFailure': {
            'task_id': 168868,
            'task_type': 'binary',
        },
        'Australian': {
            'task_id': 146818,
            'task_type': 'binary',
        },
        'bank': {
            'task_id': 14965,
            'task_type': 'binary',
        },
        'blood': {
            'task_id': 10101,
            'task_type': 'binary',
        },
        'breastTumor': {
            'task_id': 7324,
            'task_type': 'regression',
        },
        'echomonths': {
            'task_id': 7323,
            'task_type': 'regression',
        },
        'lowbwt': {
            'task_id': 7320,
            'task_type': 'regression',
        },
        'pbc': {
            'task_id': 7318,
            'task_type': 'regression',
        },
        'pharynx': {
            'task_id': 7322,
            'task_type': 'regression',
        },
        'pwLinear': {
            'task_id': 146821,
            'task_type': 'regression',
        },
        'car': {
            'task_id': 146821,
            'task_type': 'multi',
        },
        'christine': {
            'task_id': 168908,
            'task_type': 'binary',
        },
        'cnae': {
            'task_id': 9981,
            'task_type': 'multi',
        },
        'connect': {
            'task_id': 146195,
            'task_type': 'multi',
        },
        'credit': {
            'task_id': 31,
            'task_type': 'binary',
        },
        'fabert': {
            'task_id': 168852,
            'task_type': 'multi',
        },
        'Fashion': {
            'task_id': 146825,
            'task_type': 'multi',
        },
        'fried': {
            'task_id': 4885,
            'task_type': 'regression',
        },
        'Helena': {
            'task_id': 168329,
            'task_type': 'multi',
        },
        'higgs': {
            'task_id': 146606,
            'task_type': 'binary',
        },
        'house16H': {
            'task_id': 4893,
            'task_type': 'regression',
        },
        'house8L': {
            'task_id': 2309,
            'task_type': 'regression',
        },
        'houses': {
            'task_id': 5165,
            'task_type': 'regression',
        },
        'Jannis': {
            'task_id': 168330,
            'task_type': 'multi',
        },
        'jasmine': {
            'task_id': 168911,
            'task_type': 'binary',
        },
        'jungle': {
            'task_id': 167119,
            'task_type': 'multi',
        },
        'kc1': {
            'task_id': 3917,
            'task_type': 'binary',
        },
        'KDDCup09': {
            'task_id': 3945,
            'task_type': 'binary',
        },
        'kr': {
            'task_id': 3,
            'task_type': 'binary',
        },
        'mfeat': {
            'task_id': 12,
            'task_type': 'multi',
        },
        'MiniBooNE': {
            'task_id': 168335,
            'task_type': 'binary',
        },
        'mv': {
            'task_id': 4774,
            'task_type': 'regression',
        },
        'nomao': {
            'task_id': 9977,
            'task_type': 'binary',
        },
        'numerai28': {
            'task_id': 167120,
            'task_type': 'binary',
        },
        'phoneme': {
            'task_id': 9952,
            'task_type': 'binary',
        },
        'poker': {
            'task_id': 10102,
            'task_type': 'regression',
        },
        'pol': {
            'task_id': 2292,
            'task_type': 'regression',
        },
        'riccardo': {
            'task_id': 168333,
            'task_type': 'multi',
        },
        'Robert': {
            'task_id': 168332,
            'task_type': 'multi',
        },
        'segment': {
            'task_id': 146822,
            'task_type': 'multi',
        },
        'shuttle': {
            'task_id': 146212,
            'task_type': 'multi',
        },
        'sylvine': {
            'task_id': 168853,
            'task_type': 'binary',
        },
        'vehicle': {
            'task_id': 53,
            'task_type': 'multi',
        },
        'volkert': {
            'task_id': 168810,
            'task_type': 'multi',
        },
        'Albert': {
            'task_id': 189356,
            'task_type': 'binary',
        },
        'Covertype': {
            'task_id': 7593,
            'task_type': 'multi',
        },
        'guillermo': {
            'task_id': 168337,
            'task_type': 'binary',
        },
        'riccardo': {
            'task_id': 168333,
            'task_type': 'binary',
        },
        'dilbert': {
            'task_id': 168909,
            'task_type': 'multi',
        },
        'Dionis': {
            'task_id': 189355,
            'task_type': 'multi',
        },        
    }
import time
import numpy as np
import json
import os
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
try:
    from flaml import tune
except ImportError:
    from ray import tune
    print('Cannot import tune from flaml. Using tune from ray')
from flaml.model import XGBoostSklearnEstimator, LGBMEstimator
import logging
logger = logging.getLogger(__name__)

N_SPLITS = 5
RANDOM_SEED = 1
SPLIT_RATIO = 0.1 #0.33
HISTORY_SIZE = 10000000
MEM_THRES = 4*(1024**3)
SMALL_LARGE_THRES =  10000000
MIN_SAMPLE_TRAIN = 10000
MIN_SAMPLE_VAL = 10000
CV_HOLDOUT_THRESHOLD = 100000


def add_res(log_file_name, params_dic):
    with open(log_file_name, "a+") as f:  
        f.write(json.dumps(params_dic))
        f.write('\n')


class Problem:
    """The problem class which defines an AutoMLProblem task


    Methods:
        trainable_func()
    """

    def __init__(self, **kwargs):
        self._setup_search()

    def _setup_search(self):
        self._search_space = {}
        self._init_config = {}
        self._prune_attribute = None 
        self._resource_default, self._resource_min, self._resource_max = None, None, None 
        self._cat_hp_cost = {}

    @property
    def init_config(self):
        return self._init_config

    @property
    def search_space(self):
        return self._search_space

    @property
    def low_cost_partial_config(self):
        return self._low_cost_partial_config

    @property
    def cat_hp_cost(self):
        return self._cat_hp_cost

    @property
    def prune_attribute(self):
        return self._prune_attribute

    @property
    def prune_attribute_default_min_max(self):
        return self._resource_default, self._resource_min, self._resource_max  

    def trainable_func(self, config, **kwargs):
        obj = 0
        return obj


class Toy(Problem):
    """A toy problem


    Methods:
        compute_with_config(): the training function
        trainable_func(config, **kwargs): a wrapper of the training function
    """

    def __init__(self, **kwargs):
        self.name = 'toy'
        self._setup_search()

    def _setup_search(self):
        super()._setup_search()
        self._search_space = {}
        self._search_space['x'] = tune.qloguniform(1, 1000000, 1)
        self._search_space['y'] = tune.qloguniform(1, 1000000, 1) 

    def trainable_func(self, config, **kwargs):
        _, metric2minimize, time2eval = self.compute_with_config(config)
        return metric2minimize

    def compute_with_config(self, config: dict, budget_left=None, state=None):
        curent_time = time.time()
        state = None
        # should be a function of config
        metric2minimize = (round(config['x'])-95000)**2 
        time2eval = time.time() - curent_time
        return state, metric2minimize, time2eval


class AutoMLProblem(Problem):
    task = oml_tasks
    metric = {
        'binary': 'roc_auc',
        'multi': 'log_loss',
        'regression': 'r2',
    }
    data_dir = 'test/automl/'

    class BaseEstimator:
        '''The abstract class for all learners
        '''

        MEMORY_BUDGET = 80 * 1024 ** 3

        def __init__(self, objective_name='binary:logistic', n_jobs=1, **params):
            '''Constructor
            
            Args:
                objective_name: A string of the objective name, one of
                    'binary:logistic', 'multi:softmax', 'regression'
                n_jobs: An integer of the number of parallel threads
                params: A dictionary of the hyperparameter names and values
            '''
            self.params = params
            self.estimator = DummyClassifier
            self.objective_name = objective_name
            self.n_jobs = n_jobs
            self.model = None
            self._dummy_model = None

        def _size(self):
            '''the memory consumption of the model
            '''
            try:
                max_leaves = int(round(self.params['max_leaves']))
                n_estimators = int(round(self.params['n_estimators']))
            except ValueError:
                return 0    
            model_size = float((max_leaves * 3 + (max_leaves - 1) * 4 + 1) * n_estimators * 8) 
            return model_size
            
        @property
        def classes_(self):
            return self.model.classes_

        def preprocess(self, X):
            # print('base preprocess')
            return X

        def cleanup(self):
            pass 

        def __del__(self):
            self.cleanup()
     
        def dummy_model(self, X_train, y_train):
            if self._dummy_model is None:
                if self.objective_name == 'regression':
                    self._dummy_model = DummyRegressor()
                else:
                    self._dummy_model = DummyClassifier()
                self._dummy_model.fit(X_train, y_train)
            return self._dummy_model

        def fit(self, X_train, y_train, budget=None, train_full=None):
            '''Train the model from given training data
            
            Args:
                X_train: A numpy array of training data in shape n*m
                y_train: A numpy array of labels in shape n*1
                budget: time budget

            Returns:
                model: An object of the trained model, with method predict(), 
                    and predict_proba() if it supports classification
                traing_time: A float of the training time in seconds
            '''
            curent_time = time.time()
            X_train = self.preprocess(X_train)
            if self._size() > self.MEMORY_BUDGET: 
                return None, time.time() - curent_time
            model = self.estimator(**self.params)
            model.fit(X_train, y_train)
            train_time = time.time() - curent_time
            self.model = model
            return (model, train_time)

        def predict(self, X_test):
            '''Predict label from features
            
            Args:
                model: An object of trained model with method predict()
                X_test: A numpy array of featurized instances, shape n*m

            Returns:
                A numpy array of shape n*1. 
                Each element is the label for a instance
            ''' 
            X_test = self.preprocess(X_test)
            return self.model.predict(X_test)

        def predict_proba(self, X_test):
            '''Predict the probability of each class from features

            Only works for classification problems

            Args:
                model: An object of trained model with method predict_proba()
                X_test: A numpy array of featurized instances, shape n*m

            Returns:
                A numpy array of shape n*c. c is the # classes
                Each element at (i,j) is the probability for instance i to be in
                    class j
            '''
            if 'regression' in self.objective_name:
                print('Regression tasks do not support predict_prob')
                raise ValueError
            else:
                X_test = self.preprocess(X_test)
                return self.model.predict_proba(X_test)

    class LGBM_CFO(LGBMEstimator):

        MEMORY_BUDGET = 80 * 1024 ** 3

        def __init__(self, task='binary', n_jobs=1, **params):
            super().__init__(task, n_jobs, **params)
            self.params["seed"] = 9999999

        def _fit(self, X_train, y_train, **kwargs):
            if self.size(self.params) > self.MEMORY_BUDGET:
                return 0
            else:
                return super()._fit(X_train, y_train, **kwargs)

    class LGBM(LGBM_CFO):
        pass

    class LGBM_Normal(LGBM_CFO):

        @classmethod
        def search_space(cls, data_size, **params): 
            space = super().search_space(data_size, **params)
            for key in ("n_estimators", "num_leaves"):
                domain = space[key]["domain"]
                space[key]["domain"] = tune.randn(
                    np.log2(domain.lower),
                    np.log2(domain.upper / domain.lower) / 10)
            return space

        def __init__(self, task='binary', n_jobs=1, **params):
            params = params.copy()
            for key in ("n_estimators", "num_leaves"):
                params[key] = int(round(2**(2 + np.abs(params[key] - 2))))
            super().__init__(task, n_jobs, **params)

    class LGBM_MLNET(LGBM_CFO):
        @classmethod
        def search_space(cls, data_size, **params): 
            return {
                'n_estimators': {
                    'domain': tune.qloguniform(lower=2, upper=256, q=1),
                    'init_value': 2,
                },
                'max_leaves': {
                    'domain': tune.qloguniform(lower=2, upper=256, q=1),
                    'init_value': 2,
                },
                'min_data_in_leaf': {
                    'domain': tune.qloguniform(lower=2, upper=2**7, q=1),
                    'init_value': 128,
                },
                'learning_rate': {
                    'domain': tune.loguniform(lower=1e-3, upper=1.0),
                },
            }

        def __init__(self, task='binary', n_jobs=1, **params):
            super().__init__(task, n_jobs, **params)
            # Default: ‘regression’ for LGBMRegressor, 
            # ‘binary’ or ‘multiclass’ for LGBMClassifier
            self.params = {
                "n_estimators": self.params["n_estimators"],
                "max_leaves": self.params.get('max_leaves'),
                'objective': self.params.get("objective"),
                'n_jobs': n_jobs,
                'learning_rate': params["learning_rate"],
                "min_data_in_leaf": int(round(params["min_data_in_leaf"])),
            }

    class LGBM_MLNET_ALTER(LGBM_CFO):
        @classmethod
        def search_space(cls, data_size, **params): 
            return {
                'n_estimators': {
                    'domain': tune.qloguniform(lower=2, upper=256, q=1),
                    'init_value': 2,
                },
                'max_leaves': {
                    'domain': tune.qloguniform(lower=2, upper=256, q=1),
                    'init_value': 2,
                },
                'min_child_weight': {
                    'domain': tune.loguniform(lower=1e-3, upper=2**7),
                    'init_value': 2**7,
                },
                'learning_rate': {
                    'domain': tune.loguniform(lower=1e-3, upper=1.0),
                },
            }

        def __init__(self, task='binary', n_jobs=1, **params):
            super().__init__(task, n_jobs, **params)
            # Default: ‘regression’ for LGBMRegressor, 
            # ‘binary’ or ‘multiclass’ for LGBMClassifier
            self.params = {
                "n_estimators": self.params["n_estimators"],
                "max_leaves": self.params.get('max_leaves'),
                'objective': self.params.get("objective"),
                'n_jobs': n_jobs,
                'learning_rate': params["learning_rate"],
                "min_child_weight": params["min_child_weight"],
            }

    class XGB_CFO(XGBoostSklearnEstimator):

        MEMORY_BUDGET = 80 * 1024 ** 3
        def __init__(self, task='binary', n_jobs=1, **params):
            super().__init__(task, n_jobs, **params)
            self.params["seed"] = 9999999

        def _fit(self, X_train, y_train, **kwargs): 
            if self.size(self.params) > self.MEMORY_BUDGET:
                return 0
            else: 
                return super()._fit(X_train, y_train, **kwargs)

  
    class XGB_CFO_Large(XGB_CFO):
        @classmethod
        def search_space(cls, data_size, **params): 
            upper = min(32768, int(data_size))
            return {
                'n_estimators': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                    'init_value': 4,
                },
                'max_leaves': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                    'init_value': 4,
                },
                'min_child_weight': {
                    'domain': tune.loguniform(lower=1e-3, upper=2**7),
                    'init_value': 20.0,
                },
                'learning_rate': {
                    'domain': tune.loguniform(lower=2**-10, upper=1.0),
                    'init_value': 0.1,
                },
                'subsample': {
                    'domain': tune.uniform(lower=0.1, upper=1.0),
                    'init_value': 1.0,
                },                        
                'colsample_bylevel': {
                    'domain': tune.uniform(lower=0.01, upper=1.0),
                    'init_value': 1.0,
                },                        
                'colsample_bytree': {
                    'domain': tune.uniform(lower=0.01, upper=1.0),
                    'init_value': 1.0,
                },                        
                'reg_alpha': {
                    'domain': tune.loguniform(lower=2**-10, upper=2**10),
                    'init_value': 1e-10,
                },    
                'reg_lambda': {
                    'domain': tune.loguniform(lower=2**-10, upper=2**10),
                    'init_value': 1.0,
                },    
            }

    class XGB_BlendSearch(XGB_CFO):

        def __init__(self, task='binary', n_jobs=1,
                     n_estimators=4, max_leaves=4, subsample=1.0,
                     min_child_weight=1, learning_rate=0.1, reg_lambda=1.0,
                     reg_alpha=0.0, colsample_bylevel=1.0, colsample_bytree=1.0,
                     tree_method='hist', booster='gbtree', **params):
            super().__init__(task, n_jobs)
            self.params['max_depth'] = 0
            self.params = {"n_estimators": int(round(n_estimators)),
                           'max_leaves': int(round(max_leaves)),
                           'grow_policy': 'lossguide',
                           'tree_method': tree_method,
                           'verbosity': 0,
                           'nthread': n_jobs,
                           'learning_rate':float(learning_rate),
                           'subsample': float(subsample),
                           'reg_alpha': float(reg_alpha),
                           'reg_lambda': float(reg_lambda),
                           'min_child_weight': float(min_child_weight),
                           'booster': booster,
                           'colsample_bylevel': float(colsample_bylevel),
                           'colsample_bytree': float(colsample_bytree),
                           'seed': 9999999,
                            }

        @classmethod
        def search_space(cls, data_size, **params): 
            upper = min(32768, int(data_size))
            return {
                'n_estimators': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                    'init_value': 4,
                },
                'max_leaves': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                    'init_value': 4,
                },
                'min_child_weight': {
                    'domain': tune.loguniform(lower=0.001, upper=20.0),
                    'init_value': 20.0,
                },
                'learning_rate': {
                    'domain': tune.loguniform(lower=0.01, upper=1.0),
                    'init_value': 0.1,
                },
                'subsample': {
                    'domain': tune.uniform(lower=0.6, upper=1.0),
                    'init_value': 1.0,
                },                     
                'colsample_bylevel': {
                    'domain': tune.uniform(lower=0.6, upper=1.0),
                    'init_value': 1.0,
                },                    
                'colsample_bytree': {
                    'domain': tune.uniform(lower=0.7, upper=1.0),
                    'init_value': 1.0,
                },                        
                'reg_alpha': {
                    'domain': tune.loguniform(lower=1e-10, upper=1.0),
                    'init_value': 1e-10,
                },    
                'reg_lambda': {
                    'domain': tune.loguniform(lower=1e-10, upper=1.0),
                    'init_value': 1.0,
                },  
                'booster': {
                    'domain': tune.choice(['gbtree', 'gblinear']),
                },   
                'tree_method': {
                    'domain': tune.choice(['auto', 'approx', 'hist']),
                },    
            }
    
    class XGB_BS_NOINIT(XGB_BlendSearch):
        @classmethod
        def search_space(cls, data_size, **params): 
            upper = min(32768, int(data_size))
            return {
                'n_estimators': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                },
                'max_leaves': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),  
                },
                'min_child_weight': {
                    'domain': tune.loguniform(lower=0.001, upper=20.0),
                    'init_value': 20.0,
                },
                'learning_rate': {
                    'domain': tune.loguniform(lower=0.01, upper=1.0),
                    'init_value': 0.1,
                },
                'subsample': {
                    'domain': tune.uniform(lower=0.6, upper=1.0),
                    
                },                        
                'colsample_bylevel': {
                    'domain': tune.uniform(lower=0.6, upper=1.0),
                    
                },                        
                'colsample_bytree': {
                    'domain': tune.uniform(lower=0.7, upper=1.0),
                  
                },                        
                'reg_alpha': {
                    'domain': tune.loguniform(lower=1e-10, upper=1.0),
                    
                },    
                'reg_lambda': {
                    'domain': tune.loguniform(lower=1e-10, upper=1.0),
                    
                },  
                'booster': {
                    'domain': tune.choice(['gbtree', 'gblinear']),
                },   
                'tree_method': {
                    'domain': tune.choice(['auto', 'approx', 'hist']),
                },    
            }

    class XGB_BlendSearch_Large(XGB_BlendSearch):
        @classmethod
        def search_space(cls, data_size, **params): 
            upper = min(32768, int(data_size))
            return {
                'n_estimators': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                    'init_value': 4,
                },
                'max_leaves': {
                    'domain': tune.qloguniform(lower=4, upper=upper, q=1),
                    'init_value': 4,
                },
                'min_child_weight': {
                    'domain': tune.loguniform(lower=1e-3, upper=2**7),
                    'init_value': 20.0,
                },
                'learning_rate': {
                    'domain': tune.loguniform(lower=2**-10, upper=1.0),
                    'init_value': 0.1,
                },
                'subsample': {
                    'domain': tune.uniform(lower=0.1, upper=1.0),
                    'init_value': 1.0,
                },                        
                'colsample_bylevel': {
                    'domain': tune.uniform(lower=0.01, upper=1.0),
                    'init_value': 1.0,
                },                        
                'colsample_bytree': {
                    'domain': tune.uniform(lower=0.01, upper=1.0),
                    'init_value': 1.0,
                },                        
                'reg_alpha': {
                    'domain': tune.loguniform(lower=2**-10, upper=2**10),
                    'init_value': 1e-10,
                },    
                'reg_lambda': {
                    'domain': tune.loguniform(lower=2**-10, upper=2**10),
                    'init_value': 1.0,
                },   
                'booster': {
                    'domain': tune.choice(['gbtree', 'gblinear']),
                },   
                'tree_method': {
                    'domain': tune.choice(['auto', 'approx', 'hist']),
                },    
            }

    class XGB_HPOLib(XGB_CFO):
        def __init__(self, task = 'binary', n_jobs = 1, **params):
            super().__init__(task, n_jobs)
            self.params = {
            "n_estimators": int(round(params["n_estimators"])),
            'max_depth': int(round(params["max_depth"])),
            'verbosity': 0,
            'nthread': n_jobs,
            'learning_rate': params["eta"],
            'subsample': params["subsample_per_it"],
            'reg_alpha': params["reg_alpha"],
            'reg_lambda': params["reg_lambda"],
            'min_child_weight': params["min_child_weight"],
            'booster': params["booster"],
            'colsample_bylevel': params["colsample_bylevel"],
            'colsample_bytree': params["colsample_bytree"],
            'seed': 9999999,
            }

        @classmethod
        def search_space(cls, data_size, **params): 
            return {
                'n_estimators': {
                    'domain': tune.randint(lower=1, upper=256),
                    'init_value': 256,
                },
                'max_depth': {
                    'domain': tune.randint(lower=1, upper=15),
                    'init_value': 6,
                },
                'min_child_weight': {
                    'domain': tune.loguniform(lower=1, upper=2**7),
                    'init_value': 1,
                },
                'eta': {
                    'domain': tune.loguniform(lower=2**-10, upper=1.0),
                    'init_value': 0.3,
                },
                'subsample_per_it': {
                    'domain': tune.uniform(lower=0.1, upper=1.0),
                    'init_value': 1.0,
                },                        
                'colsample_bylevel': {
                    'domain': tune.uniform(lower=0.01, upper=1.0),
                    'init_value': 1.0,
                },                        
                'colsample_bytree': {
                    'domain': tune.uniform(lower=0.01, upper=1.0),
                    'init_value': 1.0,
                },                        
                'reg_alpha': {
                    'domain': tune.loguniform(lower=2**-10, upper=2**10),
                    'init_value': 1,
                },    
                'reg_lambda': {
                    'domain': tune.loguniform(lower=2**-10, upper=2**10),
                    'init_value': 1.0,
                },    
                'booster': {
                    'domain': tune.choice(['gbtree', 'gblinear', 'dart']),
                    'init_value': 'gbtree',
                }
            }

        @classmethod
        def size(cls, config):
            max_leaves = 2**int(round(config['max_depth']))
            n_estimators = int(round(config['n_estimators']))
            return (max_leaves*3 + (max_leaves-1)*4 + 1.0)*n_estimators*8
 
    class DeepTables(BaseEstimator):

        def __init__(self, objective_name='binary', n_jobs=1, **params):
            super().__init__(objective_name, n_jobs)
            self.params = params
            # assert 'epochs' in params
            # assert 'rounds' in params
            # assert 'net' in params
            self.home_dir = None

        def preprocess(self, X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[str(x) for x in list(range(
                    X.shape[1]))])
            return X

        def fit(self, X_train, y_train, budget=None, train_full=False):
            try:
                from deeptables.models.deeptable import DeepTable, ModelConfig
                from deeptables.models.deepnets import DCN, WideDeep, DeepFM
            except ImportError:
                print("pip install tensorflow==2.2.0 deeptables[gpu]")
            dropout = self.params.get('dropout', 0)
            learning_rate = self.params.get('learning_rate', 0.001)
            batch_norm = self.params.get('batch_norm', True)
            auto_discrete = self.params.get('auto_discrete', False)
            apply_gbm_features = self.params.get('apply_gbm_features', False)
            fixed_embedding_dim = self.params.get('fixed_embedding_dim', True)
            if not fixed_embedding_dim: embeddings_output_dim = 0
            else: embeddings_output_dim = 4
            stacking_op = self.params.get('stacking_op', 'add')
            if 'binary' in self.objective_name:
                # nets = DCN
                metrics, monitor = ['AUC'], 'val_auc'
            elif 'multi' in self.objective_name:
                # nets = WideDeep  
                metrics, monitor = [
                    'categorical_crossentropy'], 'val_categorical_crossentropy'
            else:
                metrics, monitor = ['r2'], 'val_r2'
            l1, l2 = 256, 128 #128, 64
            max_width = 2096
            if 'regression' != self.objective_name:
                n_classes = len(np.unique(y_train))
                base_size = max(1, min(n_classes, 100)/50)
                l1 = min(l1*base_size, max_width)
                l2 = min(l2*base_size, max_width)
            dnn_params = {'hidden_units': ((l1, dropout, batch_norm), 
            (l2, dropout, batch_norm)), 'dnn_activation': 'relu'}
            net = self.params.get('net', 'DCN')
            if net == 'DCN':
                nets = DCN
            elif net == 'WideDeep':
                nets = WideDeep
            elif net == 'DeepFM':
                nets = DeepFM
            elif net == 'dnn_nets':
                nets = [net]
            from tensorflow.keras.optimizers import Adam
            time_stamp = time.time()
            self.home_dir = f'dt_output/{time_stamp}'
            while os.path.exists(self.home_dir):
                self.home_dir += str(np.random.randint(10000000))
            conf = ModelConfig(nets=nets, earlystopping_patience=self.params['rounds'], 
                dense_dropout=self.params["dense_dropout"], 
                auto_discrete=auto_discrete, stacking_op=stacking_op,
                apply_gbm_features=apply_gbm_features,
                fixed_embedding_dim=fixed_embedding_dim,
                embeddings_output_dim=embeddings_output_dim,
                dnn_params=dnn_params,
                optimizer=Adam(learning_rate=learning_rate, clipvalue=100),
                metrics=metrics, monitor_metric=monitor, home_dir=self.home_dir)
            self.model = DeepTable(config=conf)
            log_batchsize = self.params.get('log_batchsize', 8)
            assert 'log_batchsize' in self.params.keys()
            assert 'epochs' in self.params.keys()
            self.model.fit(self.preprocess(X_train), y_train, verbose=0,
             epochs=int(round(self.params['epochs'])), batch_size=1<<log_batchsize)

        def cleanup(self):
            if self.home_dir:
                import shutil
                shutil.rmtree(self.home_dir, ignore_errors=True)
        
        @classmethod
        def search_space(cls, data_size, **params):
            early_stopping_rounds = max(min(round(1500000/data_size),150), 10)
            return {
                'rounds': {
                    'domain': tune.qloguniform(10,int(early_stopping_rounds), 1),
                    },
                'net': {
                    'domain': tune.choice(['DCN', 'dnn_nets']),
                    },
                "learning_rate": {
                    'domain': tune.loguniform(1e-4, 3e-2),
                    },
                'auto_discrete': {
                    'domain': tune.choice([False, True]),
                    },
                'apply_gbm_features': {
                    'domain': tune.choice([False, True]),
                    },
                'fixed_embedding_dim':  {
                    'domain': tune.choice([False, True]),
                    },
                'dropout': {
                    'domain': tune.uniform(0,0.5),
                    },
                'dense_dropout': {
                    'domain': tune.uniform(0,0.5),
                    },
                "log_batchsize": {
                    'domain':  8,     
                    },
                } 

    @staticmethod
    def get_estimator_from_name(name):
        if name == 'lgbm_cfo':
            estimator = AutoMLProblem.LGBM_CFO
        elif name == 'lgbm':
            estimator = AutoMLProblem.LGBM
        elif name == 'lgbm_normal_1side':
            estimator = AutoMLProblem.LGBM_Normal
        elif name == 'lgbm_mlnet':
            estimator = AutoMLProblem.LGBM_MLNET
        elif name == 'lgbm_mlnet_alter':
            estimator = AutoMLProblem.LGBM_MLNET_ALTER
        elif name == 'xgb_cfo' or ('flaml' in name and 'xgb' in name):
            estimator = AutoMLProblem.XGB_CFO
        elif name == 'xgb_cfo_large':
            estimator = AutoMLProblem.XGB_CFO_Large
        elif name in ('xgb_cat', 'xgb_blendsearch'):
            estimator = AutoMLProblem.XGB_BlendSearch
        elif name == 'xgb_blendsearch_large':
            estimator = AutoMLProblem.XGB_BlendSearch_Large
        elif name == 'xgb_bs_noinit':
            estimator = AutoMLProblem.XGB_BS_NOINIT
        elif name == 'xgb_hpolib':
            estimator = AutoMLProblem.XGB_HPOLib
        elif 'dt' in name or 'deeptable' in name:
            estimator = AutoMLProblem.DeepTables
        else: estimator = None
        return estimator

    @staticmethod
    def sklearn_metric_loss_score(metric_name, y_predict, y_true, labels = None):
        '''Loss using the specified metric

        Args:
            metric_name: A string of the mtric name, one of 
                'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'log_loss', 
                'f1', 'ap'
            y_predict: A 1d or 2d numpy array of the predictions which can be
                used to calculate the metric. E.g., 2d for log_loss and 1d
                for others. 
            y_true: A 1d numpy array of the true labels
            labels: A 1d numpy array of the unique labels
        
        Returns:
            score: A float number of the loss, the lower the better
        '''
        from sklearn.metrics import mean_squared_error, r2_score, \
            roc_auc_score, accuracy_score, mean_absolute_error, log_loss

        metric_name = metric_name.lower()
        try:
            if 'r2' in metric_name:
                score = 1.0-r2_score(y_true, y_predict)
            elif metric_name == 'rmse':
                score = np.sqrt(mean_squared_error(y_true, y_predict))
            elif metric_name == 'mae':
                score = mean_absolute_error(y_true, y_predict)
            elif metric_name == 'mse':
                score = mean_squared_error(y_true, y_predict)
            elif metric_name == 'accuracy':
                score = 1.0 - accuracy_score(y_true, y_predict)
            elif 'roc_auc' in metric_name:
                score = 1.0 - roc_auc_score(y_true, y_predict)
            elif 'log_loss' in metric_name:
                score = log_loss(y_true, y_predict, labels=labels)
            # elif 'f1' in metric_name:
            #     score = 1 - f1_score(y_true, y_predict)
            # elif 'ap' in metric_name:
            #     score = 1 - average_precision_score(y_true, y_predict)
            else:
                print('Does not support the specified metric')
                score = None
        except:
            print('score exception', metric_name)
            return np.Inf
        return score

    @staticmethod
    def generate_resource_schedule(reduction_factor, lower, upper, log_max_min_ratio = 5):
        resource_schedule = []
        if log_max_min_ratio: 
            r = max(int(upper/(reduction_factor**log_max_min_ratio)), lower)
        else: r = lower
        while r <= upper:
            resource_schedule.append(r)
            r *= reduction_factor
        if not resource_schedule:
            resource_schedule.append(upper)
        else:
            resource_schedule[-1] = upper
        print('resource_schedule', resource_schedule)
        return resource_schedule
    
     
    def _setup_search(self):
        super()._setup_search()
        try:
            space = self.estimator.search_space(self.data_size)
            self._search_space = dict((key, value['domain'])
                for key, value in space.items())

            self._init_config = dict((key, value['init_value'])
                for key, value in space.items() if 'init_value' in value)
            self._low_cost_partial_config = dict(
                (key, value['low_cost_init_value']) for key, value 
                in space.items() if 'low_cost_init_value' in value)
            print('setup_search', self._init_config, self._low_cost_partial_config)
        except:
            print('estimator', self.estimator)
            raise NotImplementedError
        if 'woinit' in self._estimator_name or 'noinit' in self._estimator_name:
            self._init_config = {}
        if 'flaml' in self._estimator_name:
            return 
        if self.estimator ==  AutoMLProblem.DeepTables:
            logger.info('setting up deeptables hpo')
            self._init_config =  {'rounds': 10}
            self._prune_attribute = 'epochs'
            #TODO: _resource_default is not necessary?
            self._resource_default, self._resource_min, self._resource_max = 2**10, 2**1, 2**10 
            self._cat_hp_cost={"net": [2,1],}
        elif self.estimator == AutoMLProblem.XGB_BlendSearch or \
            self.estimator == AutoMLProblem.XGB_BlendSearch_Large:  
            logger.info('setting up XGB_BlendSearch or XGB_BlendSearch_Large hpo')
            self._init_config =  {
                'n_estimators': 4,
                'max_leaves': 4,
                'min_child_weight': 20,
                }
            self._cat_hp_cost={
                "booster": [2, 1],
                }
        elif self.estimator == AutoMLProblem.XGB_CFO or self.estimator == AutoMLProblem.XGB_CFO_Large:
            logger.info('setting up XGB_CFO or XGB_CFO_Large hpo')
            self._init_config =  {
                'n_estimators': 4,
                'max_leaves': 4,
                'min_child_weight': 20,
                }   
        elif self.estimator in (AutoMLProblem.LGBM_CFO, AutoMLProblem.LGBM, AutoMLProblem.LGBM_Normal):
            logger.info('setting up LGBM_CFO or LGBM hpo')
            self._init_config = dict((key, value['init_value']) for key, value 
             in self.estimator.search_space(self.data_size).items()
             if 'init_value' in value)
            self._low_cost_partial_config = dict(
                (key, value['low_cost_init_value']) for key, value 
                in self.estimator.search_space(self.data_size).items()
                if 'low_cost_init_value' in value)
            # if self.estimator == AutoMLProblem.LGBM_CFO:
            #     self._init_config['min_child_weight'] = 20
            # else:
            #     self._init_config["min_data_in_leaf"] = 128
        elif self.estimator == AutoMLProblem.XGB_HPOLib:
            logger.info('setting up XGB_HPOLib hpo')
            self._init_config =  {
                'n_estimators': 1,
                'max_depth': 1,
                'min_child_weight': 2**7,
                }
        elif self.estimator in (AutoMLProblem.LGBM_MLNET, AutoMLProblem.LGBM_MLNET_ALTER):
            logger.info('setting up LGBM_MLNET hpo')
            self._init_config = dict((key, value['init_value']) for key, value 
             in self.estimator.search_space(self.data_size).items()
             if 'init_value' in value)
        else: 
            NotImplementedError
        # set the configuration (to be always the largest, assuming best at max) for hp which is prune_attribute
        if self._prune_attribute is not None:
            assert self._resource_max is not None
            self._search_space[self._prune_attribute] = self._resource_max
        
    def _get_test_loss(self, estimator = None, X_test = None, y_test = None, 
                            metric = 'r2', labels = None):
        if not estimator.model:
            loss = np.Inf
        else:
            if 'roc_auc' == metric:
                y_pred = estimator.predict_proba(X_test = X_test)
                if y_pred.ndim>1 and y_pred.shape[1]>1:
                    y_pred = y_pred[:,1]
            elif 'log_loss' == metric:
                y_pred = estimator.predict_proba(X_test = X_test)
                # print('estimator', estimator)
            elif 'r2' == metric:
                y_pred = estimator.predict(X_test = X_test)
            loss = AutoMLProblem.sklearn_metric_loss_score(metric, y_pred, y_test,
             labels)
            estimator.cleanup()
        return loss

    
    #TODO: can ray tune serise this function?
    def trainable_func(self, config, start_time, log_file_name, resource_schedule, total_budget=np.inf):
        # print('config in trainable_func', config)
        time_used = time.time() - start_time
        for epo in resource_schedule:
            loss, time2eval = self.compute_with_config(config, total_budget-time_used)
            # write result
            time_used = time.time() - start_time
            i_config = config
            if self.prune_attribute: i_config[self._prune_attribute] = epo
            log_dic = {
                'total_search_time': time_used,
                'obj': loss,
                'trial_time': time2eval,
                'config': i_config
            }
            add_res(log_file_name, log_dic)
            # TODO: how to specify the name in tune.report properly
            tune.report(epochs=epo, loss=loss)

    def compute_with_config(self, config: dict, budget_left = np.inf, state = None):
        curent_time = time.time()
        objective_name = self.objective
        metric = self.metric[objective_name]
        # print('config', config)
        estimator = self.estimator(**config, task=objective_name,
         n_jobs=self.n_jobs)
        if self.resampling_strategy == 'cv':
            total_val_loss, valid_folder_num = 0, 0 
            n = self.kf.get_n_splits()
            # print('self.y_all',self.X_all[0:5], self.y_all[0:5])
            if budget_left is not None: budget_per_train = budget_left / n
            else: budget_per_train = np.inf
            if objective_name=='regression' or True:
                labels = None
                X_train_split, y_train_split = self.X_all, self.y_all
            else:
                labels = np.unique(self.y_all) 
                l = len(labels)
                X_train_split, y_train_split = self.X_all[l:], self.y_all[l:]
            if isinstance(self.kf, RepeatedStratifiedKFold):
                kf = self.kf.split(X_train_split, y_train_split)
            else:
                kf = self.kf.split(X_train_split)
            rng = np.random.RandomState(2020)
            val_loss_list = []
            for train_index, val_index in kf:
                train_index = rng.permutation(train_index)
                if isinstance(X_train_split, pd.DataFrame):
                    X_train, X_val = X_train_split.iloc[
                        train_index], X_train_split.iloc[val_index]
                else:
                    X_train, X_val = X_train_split[train_index], X_train_split[
                        val_index]
                if isinstance(y_train_split, pd.Series):
                    y_train, y_val = y_train_split.iloc[
                        train_index], y_train_split.iloc[val_index]
                else:
                    y_train, y_val = y_train_split[
                        train_index], y_train_split[val_index] 
                # print( 'X_iclo', X_train.iloc[0:5])               
                if labels is not None:
                    X_train = AutoMLProblem.concat(self.X_all[:l], X_train)
                    y_train = np.concatenate([self.y_all[:l], y_train])
                estimator.fit(X_train, y_train, budget_per_train)
                val_loss_i = self._get_test_loss(estimator, X_val, y_val,
                 metric, self.labels)
                # train_loss = self._get_test_loss(estimator, X_train,
                #     y_train, metric, self.labels)
                # val_loss_i = 2*val_loss_i - train_loss
                try:
                    val_loss_i = float(val_loss_i)
                    valid_folder_num += 1
                    total_val_loss += val_loss_i
                    if valid_folder_num == n:
                        val_loss_list.append(total_val_loss/valid_folder_num)
                        total_val_loss = valid_folder_num = 0
                except:
                    print ('Evaluation folder failed !!!')
                    pass
            loss = np.max(val_loss_list)
        else:
            estimator.fit(self.X_train, self.y_train, budget_left)
            loss = self._get_test_loss(estimator, X_test = self.X_val,
             y_test = self.y_val, metric = metric, labels = self.labels)
            # train_loss = self._get_test_loss(estimator, X_test = self.X_train,
            #     y_test = self.y_train, metric = metric, labels = self.labels)
            # loss = 2*loss - train_loss
            # if state: state.model = estimator.model
            # print('hold out val loss', loss)
        time2eval = time.time() - curent_time
        # return state, loss, time2eval
        return loss, time2eval
    
    def get_cat_choice_org_name(self, cat_hp, choice):
        if cat_hp in self.config_search_info_cat.keys():
            choice_index = int(choice)
            choice_name = self.config_search_info[cat_hp].choices[choice_index] 
        # TODO handle not in dic error
        return choice_name

    def _decide_eval_method(self, data_shape, time_budget):
        nrow, dim = int(data_shape[0]), int(data_shape[1])
        print(nrow, dim, nrow * dim- SMALL_LARGE_THRES)
        if nrow * dim < SMALL_LARGE_THRES and nrow < CV_HOLDOUT_THRESHOLD:
            # time allows or sampling can be used and cv is necessary
            eval_method = 'cv'
        else:
            eval_method = 'holdout'
        ## always use hold
        eval_method = 'holdout'
        # print('eval method', eval_method)
        return eval_method

    @property
    def low_cost_partial_config(self):
        return self._low_cost_partial_config
    
    def __init__(self, dataset, estimator, fold, n_jobs, time_budget = None,
     resampling_strategy = None, **args): 
        
        # super().__isnit__(**args)
        self._estimator_name = estimator
        self.name = f'{dataset}-{estimator}'
        self.time_budget = time_budget
        self.transform = True
        self.n_jobs = n_jobs
        self.split_type =  "stratified"
        self.split_ratio = SPLIT_RATIO
        self.n_splits = N_SPLITS
        task = self.task[dataset]
        self.task_id, self.objective = task['task_id'], task['task_type']
        self.task_type = 'regression' if self.objective == 'regression' else \
            'classification'
        self.fold = fold
        X_all, y_all, _, _ = AutoMLProblem.load_openml_task(self.task_id, fold,
         self.task_type, self.transform)
        # X_all, y_all, self.X_test, self.y_test = AutoMLProblem.load_openml_task(task_id, fold)
        if resampling_strategy is not None: 
            self.resampling_strategy = resampling_strategy
        else: self.resampling_strategy = self._decide_eval_method(
            X_all.shape, time_budget)
        print('resampling strategy')
        self.test_loss = []
        self.X_all, self.y_all, self.X_train, self.y_train, self.X_val, \
            self.y_val, self.kf, self.labels = AutoMLProblem.split_data(
                self.task_type, self.split_type,
                self.split_ratio, self.n_splits, self.resampling_strategy, 
                X_all, y_all)
        self._X_train, self._y_train = self.X_train, self.y_train
        self.estimator = AutoMLProblem.get_estimator_from_name(estimator)
        self.data_size = len(self.y_train) if (self.y_train is not None) else int(
            len(self.y_all) * (self.n_splits-1) / self.n_splits)
        print('estimator', self.estimator)
        self._low_cost_partial_config = None
        self._setup_search()
        print('setup search space', self._search_space)

    def get_test_data(self):
        _, _, X_test, y_test = AutoMLProblem.load_openml_task(
            self.task_id, self.fold, self.task_type, self.transform)
        return self.X_all, self.y_all, X_test, y_test
        
    @staticmethod
    def load_openml_task(task_id, fold, task_type, transform):
        import os, openml, pickle
        customized_load = False
        if customized_load:
            import arff
            oml_task = openml.tasks.get_task(task_id)
            oml_dataset = oml_task.get_dataset()
            with open(oml_dataset.data_file) as f:
                ds = arff.load(f)
            train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
            split_data_train = np.asarray(ds['data'], dtype=object)[train_ind, :]
            split_data_test = np.asarray(ds['data'], dtype=object)[test_ind, :]
            predictors = [f for f in oml_dataset.features.values()
             if f.name!=oml_dataset.default_target_attribute]
            target = [f for f in oml_dataset.features.values()
             if f.name==oml_dataset.default_target_attribute]
            predictors_ind, target_ind = [p.index for p in predictors], [
                p.index for p in target]
            X_all, y_all = split_data_train[:, predictors_ind], split_data_train[:,
             target_ind]
            X_test, y_test = split_data_test[:, predictors_ind],  split_data_test[:,
             target_ind]
            print('X_all,', X_all.shape, X_all[0:5])
            from sklearn import preprocessing
            import sklearn
            le_label = preprocessing.LabelEncoder()
            le = preprocessing.OrdinalEncoder()
            # le = preprocessing.OneHotEncoder()
            # le =sklearn.pipeline.Pipeline(sklearn.preprocessing._encoders.OneHotEncoder)
            le.fit(X_all)
            X_all = le.transform(X_all)
            X_test = le.transform(X_test)
            
            le_label.fit(y_all)
            y_all = le_label.transform(y_all)
            y_test = le_label.transform(y_test)

            print('X_all_trans,', X_all.shape, X_all[0:5], y_all[0:5])
            # Encoder('label' if self.values is not None else 'no-op',
            #            target=self.is_target,
            #            encoded_type=int if self.is_target and not self.is_numerical() else float,
            #            missing_policy='mask' if self.has_missing_values else 'ignore'
            #            ).fit(self.values)
        else:
            task = openml.tasks.get_task(task_id)
            filename = 'openml_task' + str(task_id) + '.pkl'
            os.makedirs(AutoMLProblem.data_dir, exist_ok = True)
            filepath = os.path.join(AutoMLProblem.data_dir, filename)
            if os.path.isfile(filepath):
                print('load dataset from', filepath)
                with open(filepath, 'rb') as f:
                    dataset = pickle.load(f)
            else:
                print('download dataset from openml')
                dataset = task.get_dataset()
                with open(filepath, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            # X, y, cat_, _ = dataset.get_data(task.target_name, 
            #     dataset_format='array', include_ignore_attributes = True)
            X, y, cat, _ = dataset.get_data(task.target_name)
            train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0,
                    fold=fold,
                    sample=0,
                )
            if isinstance(X, pd.DataFrame):
                X_all = X.iloc[train_indices]
                y_all = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
            else:
                X_all = X[train_indices]
                y_all = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]
        # print('X_all,', X_all.shape, X_all[0:5], cat_)
        if transform:
            X_all, y_all, X_test, y_test = AutoMLProblem.transform_data(task_type,
             X_all, y_all, X_test, y_test, cat)
        # print('X_all,', type(X_all), X_all.shape, X_all[0:5])
        return X_all, y_all, X_test, y_test

    @staticmethod
    def transform_data(task_type, X, y, X_test=None, y_test=None, cat=[]):
        # from azureml.automl.runtime.featurization import data_transformer
        # transformer = data_transformer.DataTransformer(task=task_type)
        # from deeptables.models.preprocessor import DefaultPreprocessor
        # from deeptables.models.deeptable import ModelConfig
        # conf = ModelConfig(auto_encode_label=False)
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        cat_columns, num_columns = [], []
        n = X.shape[0]
        for i, column in enumerate(X.columns):
            if cat[i]:
                if X[column].nunique()==1 or X[column].nunique(
                    dropna=True)==n-X[column].isnull().sum():
                    X.drop(columns=column, inplace=True)
                    if X_test is not None:
                        X_test.drop(columns=column, inplace=True)
                    continue
                elif X[column].dtype.name == 'object':
                    X.loc[:,column].fillna('__NAN__', inplace=True)
                    if X_test is not None: 
                        X_test.loc[:,column].fillna('__NAN__', inplace=True)
                elif X[column].dtype.name == 'category':
                    current_categories = X[column].cat.categories
                    if '__NAN__' not in current_categories:
                        X.loc[:,column] = X[column].cat.add_categories(
                            '__NAN__').fillna('__NAN__')
                        if X_test is not None: 
                            X_test.loc[:,column] = X_test[
                                column].cat.add_categories('__NAN__').fillna(
                                    '__NAN__')
                cat_columns.append(column)
            else:
                # print(X[column].dtype.name)
                if X[column].nunique(dropna=True)<2:
                    X.drop(columns=column, inplace=True)
                    if X_test is not None:
                        X_test.drop(columns=column, inplace=True)
                else:
                    X.loc[:,column].fillna(np.nan, inplace=True)
                    num_columns.append(column)
        if cat_columns:
            X.loc[:,cat_columns] = X[cat_columns].astype('category', copy=False)
            if X_test is not None: 
                X_test.loc[:,cat_columns] = X_test[cat_columns].astype(
                    'category', copy=False)
        if num_columns:
            X.loc[:,num_columns] = X[num_columns].astype('float')
            transformer = ColumnTransformer([('continuous', SimpleImputer(
                missing_values=np.nan, strategy='median'), num_columns)])
            X.loc[:,num_columns] = transformer.fit_transform(X)
            if X_test is not None: 
                X_test.loc[:,num_columns] = X_test[num_columns].astype('float')
                X_test.loc[:,num_columns] = transformer.transform(X_test)
        if task_type == 'regression':
            label_transformer = None
        else:
            from sklearn.preprocessing import LabelEncoder
            label_transformer = LabelEncoder()
            y = label_transformer.fit_transform(y)
            if y_test is not None: 
                y_test = label_transformer.transform(y_test)
        return X, y, X_test, y_test

    @staticmethod 
    def split_data(task_type, split_type, split_ratio, n_splits,
     resampling_strategy, X_all, y_all):
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle
        from scipy.sparse import issparse
        if issparse(X_all): X_all = X_all.tocsr()
        X_all, y_all = shuffle(X_all, y_all, random_state=202020)        
        df = isinstance(X_all, pd.DataFrame)
        if df:
            X_all.reset_index(drop=True, inplace=True)
            if isinstance(y_all, pd.Series):
                y_all.reset_index(drop=True, inplace=True)
        kf = X_train = y_train = X_val = y_val = None     
        labels = np.unique(y_all) 
        if resampling_strategy == 'holdout':
            if task_type != 'regression':
                label_set, first = np.unique(y_all, return_index=True)
                rest = []
                last = 0
                first.sort()
                for i in range(len(label_set)):
                    rest.extend(range(last, first[i]))
                    last = first[i] + 1
                rest.extend(range(last, len(y_all)))

                X_first = X_all.iloc[first] if df else X_all[
                    first]
                X_rest = X_all.iloc[rest] if df else X_all[rest]
                y_rest = y_all.iloc[rest] if isinstance(
                    y_all, pd.Series) else y_all[rest]
                stratify = y_rest if split_type=='stratified' else None
            else:
                stratify = None
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=split_ratio,
                stratify=stratify, random_state=1)                                                                
            if task_type != 'regression':
                X_train = AutoMLProblem.concat(X_first, X_train)
                y_train = AutoMLProblem.concat(label_set,
                    y_train) if df else np.concatenate([label_set, y_train])
                X_val = AutoMLProblem.concat(X_first, X_val)
                y_val = AutoMLProblem.concat(label_set,
                    y_val) if df else np.concatenate([label_set, y_val])
        else:          
            if task_type != 'regression' and split_type == "stratified":
                print("Using StratifiedKFold")
                kf = RepeatedStratifiedKFold(n_splits= n_splits,
                    n_repeats=1, random_state=202020)
            else:
                print("Using KFold")
                kf = RepeatedKFold(n_splits= n_splits, n_repeats=1,
                    random_state=202020)
        return X_all, y_all, X_train, y_train, X_val, y_val, kf, labels

    @staticmethod
    def concat(X1, X2):
        '''concatenate two matrices vertically
        '''
        if isinstance(X1, pd.DataFrame) or isinstance(X1, pd.Series):
            df = pd.concat([X1, X2], sort=False)
            df.reset_index(drop=True, inplace=True)
            return df
        from scipy.sparse import vstack, issparse
        if issparse(X1):
            return vstack((X1, X2))
        else:
            return np.concatenate([X1, X2])


def _test_problem_parallel(problem, time_budget_s=120, n_total_pu=4,
                           n_per_trial_pu=1, method='BlendSearch',
                           log_dir_address='logs/', log_file_name='logs/example.log',
                           ls_seed=20, report_intermediate_result=False, config_BS_in_flaml=True):
    metric = 'loss'
    mode = 'min'
    resources_per_trial = {"cpu": n_per_trial_pu, "gpu": 0}   # n_per_trial_pu
    open(log_file_name, "w")
    search_space = problem.search_space
    init_config = problem.init_config
    low_cost_partial_config = problem.low_cost_partial_config
    # specify pruning config
    prune_attr = problem.prune_attribute
    default_epochs, min_epochs, max_epochs = problem.prune_attribute_default_min_max  # 2**9, 2**1, 2**10
    cat_hp_cost = problem.cat_hp_cost
    reduction_factor_asha = 4
    reduction_factor_hyperband = 3
    start_time = time.time()
    # generate resource schedule for schedulers such as ASHA, hyperband, BOHB
    if 'ASHA' in method:
        resource_schedule = problem.generate_resource_schedule(reduction_factor_asha,
                                                               min_epochs, max_epochs)
    elif 'hyberband' in method or 'BOHB' in method:
        resource_schedule = problem.generate_resource_schedule(reduction_factor_hyperband,
                                                               min_epochs, max_epochs)
    else:
        # not tuning resource dim
        resource_schedule = [max_epochs]
    min_resource = resource_schedule[0]
    max_resource = resource_schedule[-1]

    # setup the trainable function
    ## TODO: specify the trainable_func through the AutoML problem
    trainable_func = partial(problem.trainable_func, start_time=start_time,
                             resource_schedule=resource_schedule, log_file_name=log_file_name, 
                             total_budget=time_budget_s)

    # specify how many total concurrent trials are allowed
    ray.init(address="auto") #n_total_pu
    if isinstance(init_config, list):
        points_to_evaluate = init_config
    else:
        points_to_evaluate = [init_config]

    if 'BlendSearch' in method and config_BS_in_flaml:
        # the default search_alg is BlendSearch in flaml, which use Optuna as the global search learner
        # corresponding schedulers for BS are specified in flaml.tune.run
        if method == "BlendSearchNoDiscount":
            search_alg = "nodiscount"
        else:
            search_alg = None
        analysis = tune.run(
            trainable_func,
            points_to_evaluate=points_to_evaluate,
            low_cost_partial_config=low_cost_partial_config,
            cat_hp_cost=cat_hp_cost,
            metric=metric,
            mode=mode,
            prune_attr=prune_attr,
            max_resource=max_resource,
            min_resource=min_resource,
            report_intermediate_result=report_intermediate_result,
            resources_per_trial=resources_per_trial,
            config=search_space,
            local_dir=log_dir_address,
            search_alg=search_alg,
            num_samples=-1,
            time_budget_s=time_budget_s,
            use_ray=True)
    else:
        from ray.tune.suggest import BasicVariantGenerator
        scheduler = algo = None
        if 'Optuna' in method:
            from ray.tune.suggest.optuna import OptunaSearch
            import optuna
            sampler = optuna.samplers.TPESampler(seed=RANDOMSEED+int(run_index))
            algo = OptunaSearch(
                points_to_evaluate=points_to_evaluate, 
                sampler=sampler,
                space=search_space, mode=mode, metric=metric)
        elif 'CFO' in method:
            from flaml import CFO
            algo = CFO(
                metric=metric,
                mode=mode,
                space=search_space,
                points_to_evaluate=points_to_evaluate, 
                low_cost_partial_config=low_cost_partial_config,
                cat_hp_cost=cat_hp_cost,
                seed=ls_seed,
                )
        elif 'Dragonfly' in method:
            # pip install dragonfly-opt
            # doesn't support categorical
            from ray.tune.suggest.dragonfly import DragonflySearch
            algo = DragonflySearch(
                points_to_evaluate=points_to_evaluate, 
                space=search_space, mode=mode, metric=metric)
        elif 'SkOpt' == method:
            # pip install scikit-optimize
            # TypeError: '<' not supported between instances of 'Version' and 'tuple'
            from ray.tune.suggest.skopt import SkOptSearch
            algo = SkOptSearch(
                points_to_evaluate=points_to_evaluate, 
                space=search_space, mode=mode, metric=metric)
        elif 'Nevergrad' == method:
            # pip install nevergrad
            from ray.tune.suggest.nevergrad import NevergradSearch
            import nevergrad as ng
            algo = NevergradSearch(
                points_to_evaluate=points_to_evaluate, 
                space=search_space, mode=mode, metric=metric,
                optimizer=ng.optimizers.OnePlusOne)
        elif 'ZOOpt' == method:
            # pip install -U zoopt
            # ValueError: ZOOpt does not support parameters of type `Float` with samplers of type `Quantized`
            from ray.tune.suggest.zoopt import ZOOptSearch
            algo = ZOOptSearch(
                points_to_evaluate=points_to_evaluate, 
                dim_dict=search_space, mode=mode, metric=metric,
             budget=1000000)
        elif 'Ax' == method:
            # pip install ax-platform sqlalchemy
            from ray.tune.suggest.ax import AxSearch
            algo = AxSearch(
                points_to_evaluate=points_to_evaluate, 
                space=search_space, mode=mode, metric=metric)
        elif 'HyperOpt' in method:
            # pip install -U hyperopt
            from ray.tune.suggest.hyperopt import HyperOptSearch
            algo = HyperOptSearch(
                points_to_evaluate=points_to_evaluate,
                space=search_space, mode=mode, metric=metric,
                random_state_seed=RANDOMSEED + int(run_index))           

        if 'BlendSearch' in method:
            from flaml import BlendSearch
            print('low_cost_partial_config BS', low_cost_partial_config)
            algo = BlendSearch(
                points_to_evaluate=points_to_evaluate, 
                low_cost_partial_config=low_cost_partial_config,
                cat_hp_cost=cat_hp_cost,
                global_search_alg=algo,
                space=search_space, mode=mode, metric=metric, 
                seed=ls_seed,
                )
        if 'ASHA' in method:
            from ray.tune.schedulers import ASHAScheduler
            scheduler = ASHAScheduler(time_attr=prune_attr,
                                      max_t=max_resource,
                                      grace_period=min_resource,
                                      reduction_factor=reduction_factor_asha,
                                      )
            if 'ASHA' == method:
                algo = BasicVariantGenerator(points_to_evaluate=points_to_evaluate)
                scheduler = ASHAScheduler(time_attr=prune_attr,
                                          max_t=max_resource,
                                          grace_period=min_resource,
                                          reduction_factor=reduction_factor_asha,
                                          mode=mode, metric=metric
                                          )
                mode = None
                metric = None

        if 'BOHB' == method:
            from ray.tune.schedulers import HyperBandForBOHB
            from ray.tune.suggest.bohb import TuneBOHB
            algo = TuneBOHB() # points_to_evaluate=[init_config] 
            scheduler = HyperBandForBOHB(
                time_attr=prune_attr,
                max_t=max_resource,
                reduction_factor=reduction_factor_hyperband,
                )

        analysis = tune.run(trainable_func,
                            resources_per_trial=resources_per_trial,
                            config=search_space,
                            local_dir=log_dir_address,
                            num_samples=-1,
                            time_budget_s=time_budget_s,
                            verbose=1,
                            search_alg=algo
                            )

    metric = 'loss'
    mode = 'min'
    best_trial = analysis.get_best_trial(metric, mode, "all")
    loss = best_trial.metric_analysis[metric][mode]
    logger.info(f"method={method}")
    logger.info(f"dataset={oml_dataset}")
    logger.info(f"time={time.time()-start_time}")
    logger.info(f"Best model eval loss: {loss:.4f}")
    logger.info(f"Best model parameters: {best_trial.config}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', metavar='time', type=float, default=60, help="time_budget")
    parser.add_argument('-total_pu', '--n_total_pu', metavar='n_total_pu', type=int, default=26,
                        help="total number of gpu or cpu")
    parser.add_argument('-trial_pu', '--n_per_trial_pu', metavar='n_per_trial_pu', type=int, default=1,
                        help="number of gpu or cpu per trial")
    parser.add_argument('-l', '--learner_name', metavar='learner_name', type=str, default='dt', help="specify learner name")
    parser.add_argument('-r', '--run_indexes', dest='run_indexes', nargs='*', default=[0], help="The list of run indexes")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*', default= ['Optuna',], help="The method list")
        # e.g., ['Optuna', 'BlendSearch+Optuna', 'CFO', 'ASHA', 'BOHB',]
    parser.add_argument('-d', '--dataset_list', dest='dataset_list', nargs='*', default=['vehicle'], help="The dataset list")
        # e.g., ['cnae','shuttle', ] 
    parser.add_argument('-plot_only', '--plot_only', action='store_true', help='whether to only generate plots.') 
    parser.add_argument('-agg', '--agg', action='store_true', help='whether to only agg lc.')
    args = parser.parse_args()
    time_budget_s = args.time
    # NOTE: do not differentiate gpu and cpu when configuring the resource (whether pu is gpu or cpu depends your machine)
    n_total_pu = args.n_total_pu
    n_per_trial_pu = args.n_per_trial_pu
    method_list = args.method_list
    dataset_list = args.dataset_list
    run_indexes = args.run_indexes
    learner_name = args.learner_name
    cwd = os.getcwd()
    log_dir_address = cwd + f'/logs/{learner_name}/'
    fig_dir_address = cwd + f'/plots/{learner_name}/'
    os.makedirs(log_dir_address, exist_ok=True)
    os.makedirs(fig_dir_address, exist_ok=True)
    logger.addHandler(logging.FileHandler(log_dir_address + f'tune_{learner_name}.log'))
    logger.setLevel(logging.INFO)
    
    # specify the problem
    for oml_dataset in dataset_list:
        for method in method_list:
            exp_alias = f'{learner_name}_' + '_'.join(str(s) for s in [
                n_total_pu, n_per_trial_pu, oml_dataset, time_budget_s, method])
            if args.plot_only and args.agg:
                log_file_name_alias = log_dir_address + exp_alias
                get_agg_lc_from_file(log_file_name_alias, method_alias=method)
            for run_index in run_indexes:
                # get the optimization problem
                optimization_problem = AutoMLProblem(dataset=oml_dataset, estimator=learner_name, fold=int(run_index), 
                                              n_jobs=1, time_budget=None, resampling_strategy=None)
                exp_run_alias = exp_alias + '_' + str(run_index)
                log_file_name = log_dir_address + exp_run_alias + '.log'
                if args.plot_only:
                    if not args.agg:
                        plot_lc(log_file_name, name=method)
                else:
                    _test_problem_parallel(problem=optimization_problem, time_budget_s=time_budget_s,
                                           n_total_pu=n_total_pu, n_per_trial_pu=n_per_trial_pu,
                                           method=method, log_dir_address=log_dir_address, log_file_name=log_file_name,
                                           ls_seed=20 + int(run_index))
        if args.plot_only or args.agg:
            if args.agg:
                fig_alias = f'LC_{learner_name}_lc' + '_'.join(str(s) for s in [n_total_pu, n_per_trial_pu, oml_dataset, time_budget_s])
            else:
                fig_alias = f'LC_{learner_name}_lc' + '_'.join(str(s) for s in [n_total_pu, n_per_trial_pu, oml_dataset, time_budget_s, run_index])
            fig_name = fig_dir_address + fig_alias + '.pdf'
            plt.legend()
            plt.savefig(fig_name)

# example command line:
# python test/test_automl_exp.py  -t 1800.0 -l xgb_flaml_woinit -d car  -m BlendSearch_Depri -total_pu 4 -trial_pu 1  -r 3