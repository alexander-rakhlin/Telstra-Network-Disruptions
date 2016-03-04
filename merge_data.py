import pandas as pd
from os.path import join
import numpy as np

'''
log['log_feature'] ~ 386 -> log['volume']
event['event_type'] ~ 53
resource['resource_type'] ~ 10
severity, test+train: len=18552
'''
    
data_path = r'/media/torch/WD/kaggle/Telstra/'

# add neighbors features, predecessor and successor
def encode_neighbors():

    # propagate list forth or back by filling NaNs with predecessor value    
    def fill(s, rev=False):
        if rev: s = s[::-1]
        filler = 0
        for i,v in enumerate(s):
            s[i] = filler = filler if np.isnan(v) else v
        return s[::-1] if rev else s
    
    t = logFeature.reset_index().join(train, on='id').join(test, on='id',
                    lsuffix='_train', rsuffix='_test').drop_duplicates('id')

    t['n1'] = fill(t['fault_severity'].shift(1).values)
    t['n2'] = fill(t['fault_severity'].shift(-1).values, rev=True)
    
    # 'position' - post competition addition
    t['location'] = t[['location_train', 'location_test']].fillna(0).astype(int).sum(axis=1)
    groups = t.groupby('location')
    t['position'] = groups.cumcount() / groups['id'].transform(len)
   
    return t[['id', 'n1', 'n2', 'position']].set_index('id')
    

#combine all tables
resourceType = pd.read_csv(join(data_path, 'resource_type.csv'), index_col=0)
resourceTypeVectorized = pd.get_dummies(resourceType).groupby(resourceType.index).sum().astype(int)
resourceTypeVectorized['resource_type_count'] = pd.get_dummies(resourceType).groupby(resourceType.index).size()

eventType = pd.read_csv(join(data_path, 'event_type.csv'), index_col=0)
eventTypeVectorized = pd.get_dummies(eventType).groupby(eventType.index).sum().astype(int)
eventTypeVectorized['event_type_count'] = pd.get_dummies(eventType).groupby(eventType.index).size()

logFeature = pd.read_csv(join(data_path, 'log_feature.csv'), index_col=0)
logFeatureVectorized = pd.get_dummies(logFeature)
logFeatureVectorized.iloc[:, 1:] = logFeatureVectorized.iloc[:, 1:].multiply(logFeatureVectorized['volume'], axis="index")
logFeatureVectorized = logFeatureVectorized.groupby(logFeature.index).sum().astype(int)
logFeatureVectorized['log_feature_count'] = pd.get_dummies(logFeature).groupby(logFeature.index).size()

severityType = pd.read_csv(join(data_path, 'severity_type.csv'), index_col=0)
severityType['severity_type'] = [l.replace('severity_type ', '') for l in severityType['severity_type']]

train = pd.read_csv(join(data_path, 'train.csv'), index_col=0)
train['location'] = [l.replace('location ', '') for l in train['location']]
train_merged = train.join(resourceTypeVectorized).join(eventTypeVectorized).join(logFeatureVectorized).join(severityType)

test = pd.read_csv(join(data_path, 'test.csv'), index_col=0)
test['location'] = [l.replace('location ', '') for l in test['location']]
test_merged = test.join(resourceTypeVectorized).join(eventTypeVectorized).join(logFeatureVectorized).join(severityType)

neighbors = encode_neighbors()
train_merged = train_merged.join(neighbors)
test_merged = test_merged.join(neighbors)

# save
train_merged.to_csv(join(data_path, 'train_merged.csv'))
test_merged.to_csv(join(data_path, 'test_merged.csv'))