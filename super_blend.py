import pandas as pd
from os.path import join

data_path = r'/media/torch/WD/kaggle/Telstra/'

submissions = ['submission_14_locaware_tsne_f_vol_prep.csv',
                'submission_15_locaware_tsne_perplex_50_f_vol_prep.csv',
                'submission_16_blend_4_15.csv',
                'submission_17_locaware_tsne_perplex_30_f_vol_prep.csv',
                'submission_18_locaware_tsne_perplex_30_f_vol_prep.csv',
                'submission_19_locaware_tsne_perplex_30_f_vol_prep.csv',
                'submission_20_locaware_tsne_perplex_20_f_vol_prep.csv']

submission = None
for s in submissions:
    sub = pd.read_csv(join(data_path, 'submissions', s))
    if submission is None:
        submission = sub.copy()
    else:
        submission[['predict_0','predict_1','predict_2']] +=\
                                sub[['predict_0','predict_1','predict_2']]

submission[['predict_0','predict_1','predict_2']] /= len(submissions)
submission.to_csv(join(data_path, 'submissions', 'super_blend.csv'),
                  index=False)
