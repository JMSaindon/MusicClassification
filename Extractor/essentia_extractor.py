import essentia
import essentia.standard as es
import numpy as np
import pandas as pd
import os
import tqdm
import csv
import scipy.stats as stats
import math
import time


# Features avec essentia non utilisées finalement

pathTest = "../test/Test/"
pathTrain = "../train/Train/"

test_set = os.listdir(pathTest)
train_set = os.listdir(pathTrain)

if (os.path.exists("../train_data_es.csv")):
    os.remove("../train_data_es.csv")

if (os.path.exists("../test_data_es.csv")):
    os.remove("../test_data_es.csv")

trainData = open("../train_data_es.csv", "w")
testData = open("../test_data_es.csv", "w")


stats = [stats.kurtosis, stats.skew, np.max, np.min, np.mean, np.std, np.median]
failed = []


def getHeader(features):
    header = []
    for d in features.descriptorNames():
        if type(features[d]) == float:
            header.append(d)
        elif type(features[d]) == np.ndarray:
            for stat in stats:
                header.append(d + '.' + stat.__name__)
    return header

def feature_to_dict(features):
    feature_dict = dict()
    for d in features.descriptorNames():
        if type(features[d]) == float:
            feature_dict[d] = features[d]
        elif type(features[d]) == np.ndarray:
            for stat in stats:
                name = d + '.' + stat.__name__
                feature_dict[name] = stat(features[d], axis=1)
    return feature_dict



def extract(csvfile, data, path):
    features, _ = MusicExtractor()(data[0])
    features_dict = {f: [features[f]] for f in features.descriptorNames() if not isinstance(features[f], (str, np.ndarray))}

    headers = ['track_id'] + getHeader(features)

    descriptors_df = pd.DataFrame(columns=headers).set_index('track_id')
    dirty_files = []

    for filename in tqdm(data):
        f = os.path.join(path, filename)
        try:
            features, _ = MusicExtractor()(f)
            
            features_dict = feature_to_dict(features)
            features_dict["track_id"] = int(data[:-4])

            descriptors_df = descriptors_df.append(pd.DataFrame(features_dict))
        except Exception as ex:
            print(f"\033[31m{type(ex).name} {ex}\n\033[33m{f}\033[0m")
            dirty_files.append(filename)
    
    descriptors_df.to_csv(csvfile, sep=',')
    failed += dirty_files

startTrain = time.time()

extract(trainData, train_set, pathTrain)
trainData.close()

stopTrain = time.time()

startTest = time.time()

extract(testData, test_set, pathTest)
testData.close()

stopTest = time.time()

print("temps extraction train : " + str(stopTrain-startTrain))
print("temps extraction test : " + str(stopTest-startTest))
print("fichier erronés : " + ', '.join(failed))
