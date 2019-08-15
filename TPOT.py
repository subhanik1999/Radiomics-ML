import csv
import pandas as pd
import numpy as np
from sklearn import model_selection
from tpot import TPOTClassifier
import tools

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
#from pyearth import Earth
from sklearn.cross_decomposition import PLSRegression


from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skfeature.function.statistical_based import t_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF

from sklearn.feature_selection import mutual_info_classif
from skfeature.function.information_theoretical_based import LCSI
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import DISR

from scipy.stats import wilcoxon

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

file = "radiomics__T2WI(tumor)__image.csv"
# file = "radiomics__T2WI(tumor)__image.csv"
#file = "radiomics__T1C(tumor)__image.csv"

f = open(file)
csv_f = csv.reader(f)
features = next(csv_f)
dataset = pd.read_csv(file, names=features, usecols=range(1,3089), dtype=np.float64, skiprows=1, low_memory=False)
# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset.dropna(axis=1, thresh=2, inplace=True)
dataset.dropna(inplace=True)
array_OG = dataset.values
# X_og = array_OG[:,:3064]
X = array_OG[:,:3064]
# Y_og = array_OG[:,3064]
Y = array_OG[:,3064]
# Y_og = Y_og.astype('int32')
Y = Y.astype('int32')
validation_size = 0.30
seed = 7
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_og, Y_og, test_size=validation_size)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# print(Y_test)
# train = []
# for i in range(len(X_train)):
# 	row = X_train[i]
# 	row = np.append(X_train[i], Y_train[i])
# 	train.append(row)

# use_data = pd.DataFrame(data=train, dtype=np.float64)
# # print(use_data.shape)
# dataset_maj = use_data[use_data[3064] == 1]
# # print(dataset_maj.shape)
# dataset_min = use_data[use_data[3064] == 0]

# dataset_min_upsamp = resample(dataset_min, replace=True, n_samples=312)
# # dataset_maj_downsamp = resample(dataset_maj, replace=True, n_samples=72, random_state=seed)
# dataset_upsamp = pd.concat([dataset_maj, dataset_min_upsamp])
# # dataset_downsamp = pd.concat([dataset_min, dataset_maj_downsamp])
# # print(dataset_upsamp.shape)
# # print(dataset_downsamp.shape)
# array = dataset_upsamp.values
# # array = dataset_downsamp.values
# X = array[:,:3064]
# Y = array[:,3064]
# Y = Y.astype('int32')
# validation_size = 0.01
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)


pipeline_optimizer = TPOTClassifier(generations=1, population_size=2, cv=2, verbosity=2, scoring='roc_auc')
pipeline_optimizer.fit(X_train, Y_train)
Y_pred = pipeline_optimizer.predict(X_test)
print("Accuracy: " + repr(accuracy_score(Y_test, Y_pred)))
print("Average Precision Score: " + repr(average_precision_score(Y_test, Y_pred)))
print("Kappa: " + repr(cohen_kappa_score(Y_test, Y_pred)))
print("Hamming Loss: " + repr(hamming_loss(Y_test, Y_pred)))
print("AUC: " + repr(roc_auc_score(Y_test, Y_pred)))
print("Sensitivity: " + repr(recall_score(Y_test, Y_pred)))
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print("Specificity: " + repr(tn / (tn + fp)))

pipeline_optimizer.export('tpot_exported_pipeline.py')












