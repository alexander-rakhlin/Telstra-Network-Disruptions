import pandas as pd
import numpy as np
import xgboost as xgb
from os.path import join
from sklearn.cross_validation import train_test_split
from sklearn import manifold

data_path = r'/media/torch/WD/kaggle/Telstra/'

def preprocess(df):
    f = [c for c in df.columns if not c in['fault_severity', 'tsne0', 'tsne1']]
    df[f] = np.sqrt(df[f]+3.0/8.0)
    return df

def add_tsne_features(dtrain, dtest):

    tmp = dtrain.append(dtest)
    tmp = tmp.reindex_axis(dtrain.columns, axis=1) # reorder columns again
    features = tmp.columns.drop(['id', 'fault_severity'])[:39]
    features = list(features) + ['severity_type']
    tmp=tmp[features].groupby('location').mean()

    tsne = manifold.TSNE(n_components=2, verbose=2, n_iter=500, perplexity=30)
    X_2d = tsne.fit_transform(tmp.as_matrix())
    
    tmp = pd.DataFrame(tmp.reset_index(), columns=('location', 'tsne0', 'tsne1'))
    tmp[['tsne0', 'tsne1']] = X_2d
    
    dtrain = dtrain.merge(tmp, on='location', sort=False, how='left') 
    dtest = dtest.merge(tmp, on='location', sort=False, how='left')   
    
    return dtrain, dtest

test = pd.read_csv(join(data_path, 'test_merged.csv'))
train = pd.read_csv(join(data_path, 'train_merged.csv'))

train, test = add_tsne_features(train, test)

features = train.columns.drop(['id', 'fault_severity', 'location'])

mode = 'fulltrain'  # validate|fulltrain
n_classes = 3
prepr = True

params = {'objective': 'multi:softprob',
         'eval_metric': ['merror', 'mlogloss'],
         'num_class': n_classes,
         'max_depth': 5,
         'eta': 0.05,
         'sub_sample': 0.7,
         'colsample_bytree': 0.4,
         'min_child_weight': 0,
         'seed': 1,
         'silent': 1}

if prepr:
    train = preprocess(train)

X = train[features].values
y = train['fault_severity'].values

if mode == 'validate':
    print("Validating...")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=15,
                                                        stratify=y)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist  = [(dtrain,'train'), (dtest,'eval')]

    num_round = 10000
    bst = xgb.train(params, dtrain, num_round, watchlist,
                                    early_stopping_rounds=1000)
elif mode == 'fulltrain':
    print("Full training...")

    dtrain = xgb.DMatrix(X, label=y)
    watchlist  = [(dtrain,'train')]
    
    num_round = 1126
    bst = xgb.train(params, dtrain, num_round, watchlist)    
    
    print("Predicting")
    if prepr:
        test = preprocess(test)
    X_test = test[features].values
    ypred = bst.predict(xgb.DMatrix(X_test)).reshape(-1, n_classes)
    
    print("Writing submission")
    submission = pd.read_csv(join(data_path, 'sample_submission.csv'))
    cols = ['predict_0', 'predict_1', 'predict_2']
    submission[cols] = ypred
    submission.to_csv(join(data_path, 'submissions',
                           'submission.csv'), index=False)
else:
    print("Usupported mode")