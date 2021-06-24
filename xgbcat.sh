export SCREENDIR=$HOME/.screen

screen -Sdm sylvine-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d sylvine -r 0
sleep 10s
screen -Sdm sylvine-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d sylvine -r 0
sleep 10s
screen -Sdm connect-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d connect -r 0
sleep 10s
screen -Sdm connect-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d connect -r 0
sleep 10s
screen -Sdm car-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d car -r 0
sleep 10s
screen -Sdm car-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d car -r 0
sleep 10s
screen -Sdm shuttle-xgb-CFO-0 python test/test_automl_exp.py  -t 3600 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d shuttle -r 0
sleep 10s
screen -Sdm shuttle-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 3600 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d shuttle -r 0
sleep 10s
screen -Sdm christine-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d christine -r 0
sleep 10s
screen -Sdm christine-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d christine -r 0
sleep 10s
screen -Sdm Airlines-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d Airlines -r 0
sleep 10s
screen -Sdm Airlines-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d Airlines -r 0
sleep 10s
screen -Sdm Australian-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d Australian -r 0
sleep 10s
screen -Sdm Australian-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d Australian -r 0
sleep 10s
screen -Sdm blood-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d blood -r 0
sleep 10s
screen -Sdm blood-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d blood -r 0
sleep 10s
screen -Sdm Amazon-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d Amazon -r 0
sleep 10s
screen -Sdm Amazon-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d Amazon -r 0
sleep 10s
screen -Sdm adult-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d adult -r 0
sleep 10s
screen -Sdm adult-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d adult -r 0
sleep 10s
screen -Sdm Helena-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d Helena -r 0
sleep 10s
screen -Sdm Helena-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d Helena -r 0
sleep 10s
screen -Sdm phoneme-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d phoneme -r 0
sleep 10s
screen -Sdm phoneme-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d phoneme -r 0
sleep 10s
# screen -Sdm sylvine python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d sylvine -r 0 -plot_only 
# screen -Sdm connect python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d connect -r 0 -plot_only 
# screen -Sdm car python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d car -r 0 -plot_only 
# screen -Sdm shuttle python test/test_automl_exp.py  -t 3600 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d shuttle -r 0 -plot_only 
# screen -Sdm christine python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d christine -r 0 -plot_only 
# screen -Sdm Airlines python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d Airlines -r 0 -plot_only 
# screen -Sdm Australian python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d Australian -r 0 -plot_only 
# screen -Sdm blood python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d blood -r 0 -plot_only 
# screen -Sdm Amazon python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d Amazon -r 0 -plot_only 
# screen -Sdm adult python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d adult -r 0 -plot_only 
# screen -Sdm Helena python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d Helena -r 0 -plot_only 
# screen -Sdm phoneme python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d phoneme -r 0 -plot_only 


screen -Sdm credit-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d credit -r 0
sleep 10s
screen -Sdm credit-xgb-CFO-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d credit -r 1
sleep 10s
screen -Sdm credit-xgb-CFO-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d credit -r 2
sleep 10s
screen -Sdm credit-xgb-CFO-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d credit -r 3
sleep 10s
screen -Sdm credit-xgb-CFO-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d credit -r 4
sleep 10s

screen -Sdm credit-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d credit -r 0
sleep 10s
screen -Sdm credit-xgb-BlendSearch-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d credit -r 1
sleep 10s
screen -Sdm credit-xgb-BlendSearch-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d credit -r 2
sleep 10s
screen -Sdm credit-xgb-BlendSearch-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d credit -r 3
sleep 10s
screen -Sdm credit-xgb-BlendSearch-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d credit -r 4
sleep 10s

screen -Sdm credit-xgb-Optuna-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d credit -r 0
sleep 10s
screen -Sdm credit-xgb-Optuna-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d credit -r 1
sleep 10s
screen -Sdm credit-xgb-Optuna-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d credit -r 2
sleep 10s
screen -Sdm credit-xgb-Optuna-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d credit -r 3
sleep 10s
screen -Sdm credit-xgb-Optuna-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d credit -r 4
sleep 10s

screen -Sdm KDDCup09-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d KDDCup09 -r 0
sleep 10s
screen -Sdm KDDCup09-xgb-CFO-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d KDDCup09 -r 1
sleep 10s
screen -Sdm KDDCup09-xgb-CFO-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d KDDCup09 -r 2
sleep 10s
screen -Sdm KDDCup09-xgb-CFO-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d KDDCup09 -r 3
sleep 10s
screen -Sdm KDDCup09-xgb-CFO-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d KDDCup09 -r 4
sleep 10s

screen -Sdm KDDCup09-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d KDDCup09 -r 0
sleep 10s
screen -Sdm KDDCup09-xgb-BlendSearch-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d KDDCup09 -r 1
sleep 10s
screen -Sdm KDDCup09-xgb-BlendSearch-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d KDDCup09 -r 2
sleep 10s
screen -Sdm KDDCup09-xgb-BlendSearch-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d KDDCup09 -r 3
sleep 10s
screen -Sdm KDDCup09-xgb-BlendSearch-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d KDDCup09 -r 4
sleep 10s

screen -Sdm KDDCup09-xgb-Optuna-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d KDDCup09 -r 0
sleep 10s
screen -Sdm KDDCup09-xgb-Optuna-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d KDDCup09 -r 1
sleep 10s
screen -Sdm KDDCup09-xgb-Optuna-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d KDDCup09 -r 2
sleep 10s
screen -Sdm KDDCup09-xgb-Optuna-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d KDDCup09 -r 3
sleep 10s
screen -Sdm KDDCup09-xgb-Optuna-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d KDDCup09 -r 4
sleep 10s

screen -Sdm volkert-xgb-CFO-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d volkert -r 0
sleep 10s
screen -Sdm volkert-xgb-CFO-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d volkert -r 1
sleep 10s
screen -Sdm volkert-xgb-CFO-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d volkert -r 2
sleep 10s
screen -Sdm volkert-xgb-CFO-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d volkert -r 3
sleep 10s
screen -Sdm volkert-xgb-CFO-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'CFO' -l xgb_cat -d volkert -r 4
sleep 10s

screen -Sdm volkert-xgb-BlendSearch-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d volkert -r 0
sleep 10s
screen -Sdm volkert-xgb-BlendSearch-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d volkert -r 1
sleep 10s
screen -Sdm volkert-xgb-BlendSearch-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d volkert -r 2
sleep 10s
screen -Sdm volkert-xgb-BlendSearch-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d volkert -r 3
sleep 10s
screen -Sdm volkert-xgb-BlendSearch-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'BlendSearch+Optuna' -l xgb_cat -d volkert -r 4
sleep 10s

screen -Sdm volkert-xgb-Optuna-0 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d volkert -r 0
sleep 10s
screen -Sdm volkert-xgb-Optuna-1 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d volkert -r 1
sleep 10s
screen -Sdm volkert-xgb-Optuna-2 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d volkert -r 2
sleep 10s
screen -Sdm volkert-xgb-Optuna-3 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d volkert -r 3
sleep 10s
screen -Sdm volkert-xgb-Optuna-4 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m 'Optuna' -l xgb_cat -d volkert -r 4
sleep 10s

# screen -Sdm volkert python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d volkert -r 0 1 2 3 4 -plot_only -agg
# screen -Sdm KDDCup09 python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d KDDCup09 -r 0 1 2 3 4 -plot_only -agg
# screen -Sdm credit python test/test_automl_exp.py  -t 300 -trial_pu 1  -total_pu 1  -m BlendSearch+Optuna Optuna CFO  -l xgb_cat -d credit -r 0 1 2 3 4 -plot_only -agg