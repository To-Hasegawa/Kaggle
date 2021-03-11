#ref: https://www.kaggle.com/yasufuminakama/molecular-translation-naive-baseline
import pandas as pd

train = pd.read_csv('../input/bms-molecular-translation/train_labels.csv')
test = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')

def get_train_file_path(image_id):
    return "../input/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )

def get_test_file_path(image_id):
    return "../input/bms-molecular-translation/test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )

train['file_path'] = train['image_id'].progress_apply(get_train_file_path)
test['file_path'] = test['image_id'].progress_apply(get_test_file_path)