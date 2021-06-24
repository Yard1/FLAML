'''Require: 
conda create -n py37 python=3.7
pip install flaml[blendsearch,ray]
pip install tensorflow==2.2.0 deeptables[gpu]
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
import time
RANDOMSEED = 1024

try:
    import ray
    import flaml
    from flaml import BlendSearch
    from deeptables.models.deeptable import DeepTable, ModelConfig
    from deeptables.models.deepnets import DCN, WideDeep, DeepFM
except:
    print("pip install tensorflow==2.2.0 deeptables[gpu] flaml[blendsearch,ray]")
    
import logging
logger = logging.getLogger(__name__)




def construct_dt_modelconfig(config:dict, y_train, objective_name):#->ModelConfig:
    # basic config of dt
    dropout = config.get('dropout', 0)
    learning_rate = config.get('learning_rate', 0.001)
    batch_norm = config.get('batch_norm', True)
    auto_discrete = config.get('auto_discrete', False)
    apply_gbm_features = config.get('apply_gbm_features', False)
    fixed_embedding_dim = config.get('fixed_embedding_dim', True)
    if not fixed_embedding_dim: embeddings_output_dim = 0
    else: embeddings_output_dim = 4
    stacking_op = config.get('stacking_op', 'add')

    if 'binary' in objective_name:
        # nets = DCN
        metrics, monitor = ['AUC'], 'val_auc'
    elif 'multi' in objective_name:
        # nets = WideDeep  
        metrics, monitor = [
            'categorical_crossentropy'], 'val_categorical_crossentropy'
    else:
        metrics, monitor = ['r2'], 'val_r2'
    l1, l2 = 256, 128 #128, 64
    max_width = 2096
    if 'regression' != objective_name:
        n_classes = len(np.unique(y_train))
        base_size = max(1, min(n_classes, 100)/50)
        l1 = min(l1*base_size, max_width)
        l2 = min(l2*base_size, max_width)
    dnn_params = {'hidden_units': ((l1, dropout, batch_norm), 
        (l2, dropout, batch_norm)), 'dnn_activation': 'relu'}
    net = config.get('nets', 'DCN')
    if net == 'DCN':
        nets = DCN
    elif net == 'WideDeep':
        nets = WideDeep
    elif net == 'DeepFM':
        nets = DeepFM
    elif net == 'dnn_nets':
        nets = [net]
    else: nets = net
    from tensorflow.keras.optimizers import Adam
    # assert 'rounds' in config, 'rounds required'
    assert 'dense_dropout' in config, 'dense_dropout required' 
    dt_model_config = ModelConfig(nets=nets, earlystopping_patience=config[
                "rounds"], dense_dropout=config["dense_dropout"], 
                auto_discrete=auto_discrete, stacking_op=stacking_op,
                apply_gbm_features=apply_gbm_features,
                fixed_embedding_dim=fixed_embedding_dim,
                embeddings_output_dim=embeddings_output_dim,
                dnn_params=dnn_params,
                optimizer=Adam(learning_rate=learning_rate, clipvalue=100),
                metrics=metrics, monitor_metric=monitor)

    return dt_model_config, metrics[0]

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


def add_res(log_file_name, *params):
    # params =[time_used, eval_count, best_obj, best_config, choice, obj, eval_time, i_config]
    file_save = open(log_file_name, 'a+')
    line_info = '\t'.join(str(x) for x in params)
    file_save.write(line_info)
    file_save.write('\n')          

def get_test_loss(estimator = None, model=None, X_test = None, y_test = None, 
                            metric = 'r2', labels = None):
    metric = metric.lower()
    from sklearn.metrics import mean_squared_error, r2_score, \
        roc_auc_score, accuracy_score, mean_absolute_error, log_loss
    if not estimator:
        loss = np.Inf
    else:
        # if 'roc_auc' == metric:
        if 'auc' in metric:
            y_pred = estimator.predict_proba(X_test)
            if y_pred.ndim>1 and y_pred.shape[1]>1:
                y_pred = y_pred[:,1]
            loss = 1.0 - roc_auc_score(y_test, y_pred)
        elif 'log_loss' == metric or 'categorical' in metric:
            print('yes',estimator )
            att_func = getattr(estimator, "predict_proba", None)
            if callable(att_func): print('yes')
            y_pred = estimator.predict_proba(X_test)
            loss = log_loss(y_test, y_pred, labels=labels)
        elif 'r2' == metric:
            y_pred = estimator.predict(X_test)
            loss = 1.0-r2_score(y_test, y_pred)
    return loss

def train_dt(config: dict, oml_dataset:str, start_time: float, prune_attr: str, 
        resource_schedule: list, log_file_name: str):
    """ implement the traininig function of dt model
        reference: blendsearch.problem.DeepTables
    """

    def preprocess(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[str(x) for x in list(range(
                X.shape[1]))])
        return X

    # get a multi-class dataset
    from sklearn.model_selection import train_test_split
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(name=oml_dataset, return_X_y=True)
        logger.info(f"dataset={oml_dataset}")
    except:
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        logger.info("failed to fetch oml dataset, dataset=wine")

    #NOTE: only considering classification dataset
    # assert len(set(y)) >2, 'only considering classification dataset'
    if len(set(y)) >2: 
        objective_name = 'multi'
    else: objective_name = 'binary'
    # objective_name = 'multi'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
        random_state=42)
    # setup deeptable learner
    from deeptables.models.deeptable import DeepTable, ModelConfig
    from deeptables.models.deepnets import DCN, WideDeep, DeepFM

    # clip 'rounds' according to the size of the data
    data_size = len(y_train)
    if prune_attr in config: resource_schedule = [config[prune_attr]]
    config = config.copy()
    for epo in resource_schedule:
        if 'rounds' not in config:
            # assert 'epochs' in config
            config['rounds'] = int(round(epo))
        else:
            # assert 'rounds' in config, 'rounds required'
            clipped_rounds = max(min(round(1500000/data_size),round(config['rounds'])), 10)
            config['rounds'] = int(clipped_rounds)
        dt_model_config, dt_metric = construct_dt_modelconfig(config, y_train, objective_name=objective_name)
        dt = DeepTable(dt_model_config)
        log_batchsize = config.get('log_batchsize', 9)
        # train model 
        eval_start_time = time.time()
        dt_model, _ = dt.fit(preprocess(X_train), y_train, verbose=0,
                epochs=int(round(epo)), batch_size=1<<log_batchsize)

        # evaluate model
        # DT_loss_metric = 'categorical_crossentropy'
        loss = get_test_loss(dt, dt_model, X_test, y_test, metric=dt_metric)

        # write result
        eval_time = time.time() - eval_start_time 
        time_used = time.time()-start_time
        # NOTE: these fields are missing
        eval_count, best_obj, best_config, choice = None, None, None, None  # missing fields
        obj = loss 
        i_config = config
        i_config[prune_attr] = epo
        log_param = [time_used, eval_count, best_obj, best_config, choice, obj, eval_time, i_config]
        add_res(log_file_name, log_param)

        # report loss
        tune.report(epochs=epo, loss=loss)
        

def _test_dt_parallel(time_budget_s= 120, n_total_pu=4, n_per_trial_pu=1, method='BlendSearch', run_index=0, \
    oml_dataset = 'shuttle', log_dir_address = '/home/qiw/FLAML/logs/', log_file_name='/home/qiw/FLAML/logs/example.log'):
    metric = 'loss'
    mode = 'min'
    resources_per_trial = {"cpu":n_per_trial_pu, "gpu":0 } #n_per_trial_pu
    try:
        import ray
    except ImportError:
        return
    # if 'BlendSearch' in method:
    #     from flaml import tune
    # else:
    #     from ray import tune
    from ray import tune
    # specify exp log file
    open(log_file_name,"w")
    # set random state
    # np.random.seed(RANDOMSEED)
    # specify the search space
    search_space = {
        'rounds': tune.qloguniform(10,150, 1),
        'net': tune.choice(['DCN', 'dnn_nets']),
        "learning_rate": tune.loguniform(1e-4, 3e-2),
        'auto_discrete': tune.choice([False, True]),
        'apply_gbm_features': tune.choice([False, True]),
        'fixed_embedding_dim': tune.choice([False, True]),
        'dropout': tune.uniform(0,0.5),
        'dense_dropout': tune.uniform(0,0.5),
        # "log_batchsize": tune.randint(4, 10),   
        "log_batchsize": 8,     
    }   
    # specify the init config
    init_config = {
        'rounds': 10,
        # "log_batchsize": 9,  
    }

    # specify pruning config
    prune_attr='epochs'
    default_epochs = 2**9
    min_epochs = 2**1
    max_epochs = 2**10
    reduction_factor_asha = 4
    reduction_factor_hyperband = 3
    start_time = time.time()
    report_intermediate_result = True
    if 'ASHA' in method:
        resource_schedule = generate_resource_schedule(reduction_factor_asha, 
            min_epochs, max_epochs)
    elif 'hyberband' in method or 'BOHB' in method:
        resource_schedule=  generate_resource_schedule(reduction_factor_hyperband, 
            min_epochs, max_epochs)
    else: 
        resource_schedule = [default_epochs]
        # if no pruner, add `epochs` as part of the search space
        search_space['epochs'] =  tune.loguniform(2**1,2**10)  
        # TODO: should we add 'epochs' as part of the init_config?
        # init_config['epochs'] = 2**1
        # report_intermediate_result = False

        search_space['epochs'] =  2**10

    min_resource = resource_schedule[0]
    max_resource = resource_schedule[-1]

    # create trainable function
    trainable_func = partial(train_dt, oml_dataset=oml_dataset, start_time=start_time, prune_attr=prune_attr,
    resource_schedule=resource_schedule, log_file_name=log_file_name)

    ray.init(num_cpus=n_total_pu, num_gpus=0) #n_total_pu
    if 'BlendSearch' in method and False:
        # the default search_alg is BlendSearch in flaml 
        # corresponding schedulers for BS are specified in flaml.tune.run
        analysis = tune.run(
            trainable_func,
            init_config = init_config,
            cat_hp_cost={
                "net": [2,1],
            },
            metric=metric, 
            mode=mode,
            prune_attr=prune_attr,
            max_resource=max_resource,
            min_resource=min_resource,
            report_intermediate_result=report_intermediate_result,
            resources_per_trial=resources_per_trial,
            config=search_space, 
            local_dir=log_dir_address,
            num_samples=-1, 
            time_budget_s=time_budget_s,
            use_ray=True) 
    else: 
        algo = None
        scheduler = None
        if 'Optuna' in method:
            from ray.tune.suggest.optuna import OptunaSearch
            import optuna
            sampler = optuna.samplers.TPESampler(seed=RANDOMSEED+int(run_index))
            algo = OptunaSearch(sampler=sampler)
        elif 'CFO' in method:
            from flaml import CFO
            algo = CFO(
                metric=metric,
                mode=mode,
                space=search_space,
                points_to_evaluate=[init_config], 
                cat_hp_cost={
                "net": [2,1],
                    }
                )
        # 'BlendSearch+Optuna',  'BlendSearch'
        if 'BlendSearch' in method:
            from flaml import BlendSearch

            algo = BlendSearch(
                # metric=metric,
                # mode=mode,
                # space=search_space,
                points_to_evaluate=[init_config], 
                cat_hp_cost={
                "net": [2,1],
                },
                global_search_alg=algo,
                )

        if 'ASHA' in method:
            from ray.tune.schedulers import ASHAScheduler
            scheduler = ASHAScheduler(
                time_attr=prune_attr,
                max_t=max_resource,
                grace_period=min_resource,
                reduction_factor=reduction_factor_asha,
                )
        # else: scheduler = None

        if 'BOHB' == method:
            from ray.tune.schedulers import HyperBandForBOHB
            from ray.tune.suggest.bohb import TuneBOHB
            algo = TuneBOHB() 
            scheduler = HyperBandForBOHB(
                time_attr=prune_attr,
                max_t=max_resource,
                reduction_factor=reduction_factor_hyperband,
                )
        analysis = tune.run(
            trainable_func,
            metric=metric, 
            mode=mode,
            resources_per_trial=resources_per_trial,
            config=search_space, 
            local_dir=log_dir_address,
            num_samples=-1, 
            time_budget_s=time_budget_s,
            scheduler=scheduler, 
            search_alg=algo)
    ray.shutdown()

    best_trial = analysis.get_best_trial(metric,mode,"all")
    # accuracy = 1. - best_trial.metric_analysis["eval-error"]["min"]
    logloss = best_trial.metric_analysis[metric][mode]
    logger.info(f"method={method}")
    logger.info(f"dataset={oml_dataset}")
    logger.info(f"time={time.time()-start_time}")
    logger.info(f"Best model eval loss: {logloss:.4f}")
    logger.info(f"Best model parameters: {best_trial.config}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    from visualization import get_agg_lc_from_file, plot_lc
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time', metavar='time', type = float, 
        default=60, help="time_budget")
    parser.add_argument('-total_pu', '--n_total_pu', metavar='n_total_pu', type = int, 
        default=26, help="total number of gpu or cpu")
    parser.add_argument('-trial_pu', '--n_per_trial_pu', metavar='n_per_trial_pu', type = int, 
        default=1, help="number of gpu or cpu per trial")
    parser.add_argument('-r', '--run_indexes', dest='run_indexes', nargs='*' , 
        default= [0], help="The list of run indexes")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*' , 
        default= ['BlendSearch+Optuna', 'Optuna', 'BlendSearch+Optuna+ASHA',   'Optuna+ASHA',], help="The method list") #'BOHB',
    # 1. verify whether should we fix #epo or add #epo in search space
    # 2. run sequential exp (use setting according to 1): ['BlendSearch+Optuna', 'Optuna', 'CFO'] 
    # para: 
    # 1. verify ['BlendSearch+Optuna', 'BlendSearch+Optuna+ASHA', ]
    # ['BlendSearch+Optuna', 'Optuna', 'BlendSearch+Optuna+ASHA',   'ASHA',  'Optuna+ASHA', 'BOHB',]
    parser.add_argument('-d', '--dataset_list', dest='dataset_list', nargs='*' , 
        default= ['vehicle'], help="The dataset list") # ['cnae','shuttle', ] 
    parser.add_argument('-plot_only', '--plot_only', action='store_true',
                        help='whether to only generate plots.') 
    parser.add_argument('-agg', '--agg', action='store_true',
                        help='whether to only agg lc.')
#     #TODO: exp on 'cane' has error: 
#     # File "/home/qiw/miniconda3/envs/py37/lib/python3.7/site-packages/sklearn/compose/_column_transformer.py", 
#     # line 562, in transform  "Given feature/column names do not match the ones for the "
    
#     # NOTE: the following does not work
#     # find venv/lib/python3.7/site-packages/deeptables/preprocessing/transformer.py
#     # comment line 39:
#     # y = y.argmax(axis=-1)
    args = parser.parse_args()
    time_budget_s = args.time
    # NOTE: do not differentiate gpu and cpu when configuring the resource (whether pu is gpu or cpu depends your machine)
    n_total_pu = args.n_total_pu
    n_per_trial_pu = args.n_per_trial_pu
    method_list = args.method_list
    dataset_list = args.dataset_list
    run_indexes = args.run_indexes
    cwd = os.getcwd()
    log_dir_address = cwd + '/logs/dt/'
    if not os.path.isdir(log_dir_address): os.mkdir(log_dir_address)
    logger.addHandler(logging.FileHandler(log_dir_address+'tune_dt_para.log'))
    logger.setLevel(logging.INFO)
    
    for oml_dataset in dataset_list:
        for method in method_list:
            exp_alias = 'dt_parallel_' + '_'.join(str(s) for s in [n_total_pu, n_per_trial_pu, oml_dataset, time_budget_s, method])
            if args.plot_only and args.agg:
                log_file_name_alias =  log_dir_address + exp_alias
                get_agg_lc_from_file(log_file_name_alias, method_alias=method)
            for run_index in run_indexes:
                exp_run_alias = exp_alias + '_' + run_index
                log_file_name = log_dir_address + exp_run_alias + '.log'
                if args.plot_only:
                    if not args.agg: plot_lc(log_file_name,  name=method) #y_min=1e-1,y_max=0.5,
                    else: pass
                else:
                    _test_dt_parallel(time_budget_s= time_budget_s, n_total_pu= n_total_pu, n_per_trial_pu=n_per_trial_pu, \
                        method=method, run_index=run_index, oml_dataset=oml_dataset, log_dir_address=log_dir_address, log_file_name=log_file_name)
        if args.agg:
            fig_alias = 'LC_dt_parallel_lc' + '_'.join(str(s) for s in [n_total_pu, n_per_trial_pu, oml_dataset, time_budget_s])
        else:
            fig_alias = 'LC_dt_parallel_lc' + '_'.join(str(s) for s in [n_total_pu, n_per_trial_pu, oml_dataset, time_budget_s, run_index])
        fig_name = log_dir_address + fig_alias + '.pdf'
        plt.legend()
        plt.savefig(fig_name)

# python test/test_dt_parallel.py -n 5 -t 3600 -total_pu 4

# python test/test_dt_parallel.py -n 1 -t 60 -trial_pu 4 -m 'BlendSearch+ASHA' -plot_only
# python test/test_dt_parallel.py -n 1 -t 60 -trial_pu 4 -total_pu 12
# python test/test_dt_parallel.py -n 1 -t 300 -trial_pu 2 -m 'CFO+ASHA'