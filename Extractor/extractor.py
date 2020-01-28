import librosa
import librosa.feature as feat
import numpy as np
import os
import csv
import pandas as pd
import scipy.stats as stats
import math
import time

import matplotlib.pyplot as plt
import librosa.display

# Donner les répertoires dans lesquels se trouvent les musiques de test et de train
pathTest = "../test/Test/"
pathTrain = "../train/Train/"

test_set = os.listdir(pathTest)
train_set = os.listdir(pathTrain)

# fichiers créés pour stocker les features de test et de train
traincsv = "../train_data.csv"
testcsv = "../test_data.csv"

if (os.path.exists(traincsv)):
    os.remove(traincsv)

if (os.path.exists(testcsv)):
    os.remove(testcsv)

trainData = open(traincsv, "w")
testData = open(testcsv, "w")

stats = [stats.kurtosis, stats.skew, np.max, np.min, np.mean, np.std, np.median]
features1 = [feat.chroma_cens, feat.chroma_cqt, feat.chroma_stft, feat.mfcc, feat.spectral_bandwidth, feat.spectral_centroid, feat.spectral_contrast, feat.spectral_rolloff, feat.tonnetz, feat.poly_features]
features2 = [feat.rms, feat.spectral_flatness, feat.zero_crossing_rate]

failed = []

def arrayToString(array, sep):
    s = str(array[0])
    for val in array[1:]:
        s += sep + str(val)
    return s

def extractFeature(csvfile, data, path):
    writer = csv.writer(csvfile, delimiter=',')
    print("Début extraction")
    count = 1
    print(str(count) + '/' + str(len(data)))
    header = ['track_id']
    entry = data[0]
    s = np.array([entry[:-4]])
    y, sr = librosa.load(path + entry, sr=44100)
    for feat in features1:
        featRes = feat(y=y, sr=sr)
        for stat in stats:
            st = stat(featRes, axis=1)
            s = np.concatenate((s, st))
            header += [feat.__name__ + '.' + str(i+1) + '.' + stat.__name__ for i in range(st.size)]

    for feat in features2:
        featRes = feat(y=y)
        for stat in stats:
            st = stat(featRes, axis=1)
            s = np.concatenate((s, st))
            header += [feat.__name__ + '.' + str(i+1) + '.' + stat.__name__ for i in range(st.size)]
    writer.writerow(header)
    writer.writerow(s)

    for entry in data[1:]:
        count += 1
        print(str(count) + '/' + str(len(data)))
        s = np.array([entry[:-4]])
        try:
            y, sr = librosa.load(path + entry, sr=44100)
        except:
            failed.append(entry)
            continue # skip this entry
        for feat in features1:
            featRes = feat(y=y, sr=sr)
            for stat in stats:
                st = stat(featRes, axis=1)
                s = np.concatenate((s, st))

        for feat in features2:
            featRes = feat(y=y)
            for stat in stats:
                st = stat(featRes, axis=1)
                s = np.concatenate((s, st))
        writer.writerow(s)

startTrain = time.time()

extractFeature(trainData, train_set, pathTrain)
trainData.close()

stopTrain = time.time()

startTest = time.time()

extractFeature(testData, test_set, pathTest)
testData.close()

stopTest = time.time()

print("temps extraction train : " + str(stopTrain-startTrain))
print("temps extraction test : " + str(stopTest-startTest))
print("fichier erronés : " + ', '.join(failed))
