# Kaggle's Telstra Network Disruptions challenge

This solution scored 0.46755 and achieved 81st place out of 974 (top 10%) on the [Private Leaderboard](https://www.kaggle.com/c/telstra-recruiting-network/leaderboard) of the Kaggle ["Telstra Network Disruptions"](https://www.kaggle.com/c/telstra-recruiting-network) competition. Post competition it was improved to 0.43349 (see 'position' in **merge_data.py**)

## Dependencies

* The [XGBoost](https://github.com/dmlc/xgboost) library should be installed
* The standard Python packages **numpy**, **pandas** and **sklearn** are required
* The competition datasets (the files **sample_submission.csv**, **severity_type.csv**, **train.csv**, **test.csv**, **log_feature.csv**, **resource_type.csv** and **event_type.csv**) can be downloaded from [here](https://www.kaggle.com/c/telstra-recruiting-network/data)

## How to generate the solution

 1. Set up your paths in **merge_data.py** and **model_xgboost.py** and run **merge_data.py** to create merged data files on your disk
 2. Run **model_xgboost.py** and let it generate submission
 3. Additionally, you can generate as many submissions as you like and blend them with **super_blend.py**
