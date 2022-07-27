import numpy as np
import json
import sys
import jsonpickle


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef

from scipy import spatial

import random
random.seed(1337)
entries = json.load(open(sys.argv[1]))
random.shuffle(entries)
smiles = set()
new_entries = []
colors = set()
level = "d1_colors"

external_test = []
training = []
for i, entry in enumerate(entries):
    if level in entry:
        if entry["smiles"] not in smiles:
            new_entries.append(entry)
            for c in entry[level]:
                colors.add(c)
            smiles.add(entry["smiles"])
            training.append(entry)
        else:
            external_test.append(entry)

from sklearn.metrics.pairwise import cosine_similarity
import heapq

print("cosining")
# print(cosine_similarity([x["spectrum"] for x in external_test], [x["spectrum"] for x in training]))

# similarities = cosine_similarity([x["spectrum"] for x in external_test], [x["spectrum"] for x in training])
# print(similarities.shape)
# for i, entry in enumerate(external_test):
#    print(entry["Name"])
#    indices = sorted(range(similarities.shape[1]), key=lambda x: similarities[i][x])[-3:][::-1]
#    for k in indices:
#        print("\t", training[k]["Name"], similarities[i][k])


# print(len(external_test), len(smiles))
# exit()

import pandas as pd
entries = new_entries
spectra = np.array([e["spectrum"] for e in entries], dtype=np.int64)
X = spectra
#X = spectra[:,~np.all(spectra == 0, axis =0)]
#X = spectra[:, np.apply_along_axis(np.count_nonzero, 0, spectra) >= 100]
print(X.shape)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.preprocessing import StandardScaler

sss = StratifiedKFold(n_splits=5)
color_results = {}

#json.dump({"training": training, "holdout": external_test}, open("./for_pipeline/training_holdout_data.json", 'a+'))

from sklearn.feature_selection import SelectKBest, chi2, SelectFwe, SelectFromModel, SequentialFeatureSelector
from sklearn.ensemble import ExtraTreesClassifier

for i, color in enumerate(colors):
    if False: #color == "C((C,1),(C,1),(C,2))":
        pass
    else:
        num_trees = 100
        depth = 30
        n_jobs = 7
        for num_trees in [100]:
            print(color, i, len(colors), num_trees)
            if True:
                y = np.array([1 if color in e[level] else 0 for e in entries], dtype=np.int8)
                if sum(y) > 1000:
                    transformer = SelectFwe(alpha=.05/X.shape[1])
                    transformer.fit(X,y)
                    X_new = transformer.transform(X)
                    #X_new = SelectFwe(alpha=.05/X.shape[1]).fit_transform(X, y)
                    print(X_new.shape)
                    clf = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced_subsample", n_estimators=num_trees, max_depth=depth)
                    precisions = []
                    TPs = []
                    FPs = []
                    TNs = []
                    FNs = []
                    recalls = []
                    MCCs = []
                    for train_index, test_index in sss.split(X, y):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        clf.fit(X_train, y_train)
                        # for mz, x in enumerate(clf.feature_importances_):
                        #    print("mz: ", mz, x)
                        score = clf.score(X_test, y_test)
                        pred = clf.predict(X_test)
                        true_positive = 0
                        true_negative = 0
                        false_positive = 0
                        false_negative = 0
                        mcc = matthews_corrcoef(y_test, pred)

                        for predicted, truth in zip(pred, y_test):
                            if predicted == truth and truth == 1:
                                true_positive += 1
                            elif predicted == truth and truth == 0:
                                true_negative += 1
                            elif predicted != truth and truth == 1:
                                false_negative += 1
                            else:
                                false_positive += 1
                        try:
                            precision = true_positive / (true_positive + false_positive)
                        except:
                            precision = 0
                        try:
                            recall = true_positive / (true_positive + false_negative)
                        except:
                            recall = 0
                        precisions.append(precision)

                        TPs.append(true_positive)
                        FPs.append(false_positive)
                        TNs.append(true_negative)
                        FNs.append(false_negative)
                        MCCs.append(mcc)
                        recalls.append(recall)
                        print("\t", color, "MCC: ", round(mcc, 3), " Score: ", round(score, 3), " Precision: ",
                              round(precision, 3), " Recall: ", round(recall, 3), " FN: ", false_negative, " FP: ",
                              false_positive, " TP: ", true_positive, " TN: ", true_negative)
                        break
                    print("\t\t", np.mean(precisions), np.mean(TPs), np.mean(FPs), np.mean(MCCs), "\n")
                    color_results[color] = {"precision: ": np.mean(precisions),
                                            "recall: ": np.mean(recalls),
                                            "TPs: ": np.mean(TPs),
                                            "FPs: ": np.mean(FPs),
                                            "TNs: ": np.mean(TNs),
                                            "FNs: ": np.mean(FNs)}

                    clf_new = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced_subsample",
                                                     n_estimators=num_trees, max_depth=depth)
                    model = clf_new.fit(X_new,y)
                    model = jsonpickle.encode(model)
                    transformer = jsonpickle.encode(transformer)

                    model_results = {
                        "color": color,
                        "transformer": transformer,
                        "precision": np.mean(precisions),
                        "recall": np.mean(recalls),
                        "TPs": np.mean(TPs),
                        "FPs": np.mean(FPs),
                        "TNs": np.mean(TNs),
                        "FNs": np.mean(FNs),
                        "model": model
                    }
                    json.dump(model_results, open("./for_pipeline/d1/" + color + "_" + str(depth) + "d.json", 'a+'))


#similarities = cosine_similarity([x["spectrum"] for x in external_test], [x["spectrum"] for x in training])
# print(similarities.shape)
#for i, entry in enumerate(external_test):
#   print(entry["Name"])
#   indices = sorted(range(similarities.shape[1]), key=lambda x: similarities[i][x])[-3:][::-1]
#   for k in indices:
#       print("\t", training[k]["Name"], similarities[i][k])



# json.dump(color_results, open("results_d1_rf.json", 'a+'), indent=4, sort_keys=True)
# json.dump(model_results, open("modelds_d1_rf.json", 'a+'), indent=4, sort_keys=True)
