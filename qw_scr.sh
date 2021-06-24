# git fetch upstream
# git merge xxx

# screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  'BlendSearch+Optuna+fixedepo' -r  0  -plot_only

# screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  0 
# sleep 10s
# screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 1
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  2
# sleep 10s
# screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 3
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  4
# sleep 10s


# screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo' -r  0 
# sleep 10s
# screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 1
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  2
# sleep 10s
# screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 3
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  4
# sleep 10s

# sleep 2000s
# screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo' -r  0
# sleep 10s
# screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 1
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  2
# sleep 10s
# screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 3
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  4
# sleep 10s


# screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo' -r  0
# sleep 10s
# screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 1
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  2
# sleep 10s
# screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 3
# sleep 10s
# screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  4
# sleep 10s

screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r  5
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r 6
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r 7
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r 8
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r  9
sleep 10s

screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  5
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 6
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 7
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 8
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  9
sleep 10s

sleep 2000s
screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo' -r  5 
sleep 10s
screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 6
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  7
sleep 10s
screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 8
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  9
sleep 10s


screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r  5
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r 6
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r 7
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r 8
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r  9
sleep 10s

sleep 2000s
screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo' -r  5
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 6
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  7
sleep 10s
screen -Sdm Optuna+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r 8
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'Optuna+fixedepo'  -r  9 
sleep 10s


screen -Sdm BS python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo' -r  5
sleep 10s
screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 6
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  7
sleep 10s
screen -Sdm CFO+fixedepo python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r 8
sleep 10s
screen -Sdm ASHA python test/test_dt_parallel.py  -t 1800 -d 'vehicle' -trial_pu 2 -total_pu 2  -m 'CFO+fixedepo'  -r  9
sleep 10s

python test/test_dt_parallel.py  -t 1800 -d 'phoneme' -trial_pu 2 -total_pu 2  -m 'BlendSearch+Optuna+fixedepo'  -r  5



python test/test_automl_exp.py  -t 600 -trial_pu 2  -total_pu 2  -m 'BlendSearch+Optuna' -r 0 -d 'phoneme'