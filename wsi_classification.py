import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import sys
import os
import numpy as np
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../../'))
parser = argparse.ArgumentParser(description='wsi classification')
parser.add_argument('--probs_map_features_train', default=None, metavar='TRAIN_PATH', type=str, help='Path of the training probmap features')
parser.add_argument('--probs_map_features_val', default=None, metavar='VAL_PATH', type=str, help='Path of the validating probmap features')
parser.add_argument('--probs_map_features_test', default=None, metavar='TEST_PATH', type=str, help='Path of the testing probmap features')

parser.add_argument('--logName', default=None, type=str, help='Name of the log dir')
parser.add_argument('--backbone', default='rf', metavar='BACKBONE', type=str,
                    help='The backbone module to choose, rf/knn/svm/gnb')
parser.add_argument('--feature_start_index', default=0, type=int, help='the start index of the features')

args = parser.parse_args()

FEATURE_START_INDEX = args.feature_start_index

# 创建文件保存文件夹
curr_path = os.getcwd()
save_path = os.path.join(curr_path, 'results', args.logName)

def plot_roc(gt_y, prob_predicted_y):
    predictions = prob_predicted_y[:, 1]
    fpr, tpr, _ = roc_curve(gt_y, predictions)
    np.save(os.path.join(save_path, "slide_fpr.npy"), fpr)
    np.save(os.path.join(save_path, "slide_tpr.npy"), tpr)


def validate(x, y, clf):
    predicted_y = clf.predict(x)
    prob_predicted_y = clf.predict_proba(x)
    logging.info('confusion matrix:')
    logging.info(pd.crosstab(y, predicted_y, rownames=['Actual'], colnames=['Predicted']))

    return predicted_y, prob_predicted_y


def train(x, y, backbone):
    if backbone == 'rf':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
        clf.fit(x, y)
    elif backbone == 'knn':
        # n=5
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
        clf.fit(x, y)
    elif backbone == 'svm':
        # 线性核
        clf = svm.SVC(kernel='linear', C=1.5, probability=True)
        clf.fit(x, y)
    elif backbone == 'gnb':
        clf = GaussianNB()
        clf.fit(x, y)

    return clf


def data_load(data_path):
    # 从csv中读取data frame格式的数据
    df = pd.read_csv(data_path)

    # 获取特征数（包括标签）
    features = len(df.columns)
    # 获取名字
    feature_names = df.columns[FEATURE_START_INDEX:features - 1]
    label_name = df.columns[features - 1]

    return df[feature_names], df[label_name]



def run():
    # 训练及验证数据加载
    train_x, train_y = data_load(args.probs_map_features_train)
    test_x, test_y = data_load(args.probs_map_features_test)

    model = train(train_x, train_y, args.backbone)
    predict_y, prob_predict_y = validate(test_x, test_y, model)
    plot_roc(test_y, prob_predict_y)

def main():
    logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()

