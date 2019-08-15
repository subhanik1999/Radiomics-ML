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
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

file = "radiomics__(tumor)__image_MALVBEN.csv"
file_t = "malVBen_trainOnly.csv"
f = open(file_t)
csv_f = csv.reader(f)
features = next(csv_f)
dataset = pd.read_csv(file_t, names=features, usecols=range(1,3089), dtype=np.float64, skiprows=1, low_memory=False)
dataset.dropna(axis=1, thresh=2, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset[np.isfinite(dataset).all(1)]
array_train = dataset.values

fi = open(file)
csv_fi = csv.reader(fi)
features = next(csv_fi)
dataset = pd.read_csv(file, names=features, usecols=range(1,3089), dtype=np.float64, skiprows=1, low_memory=False)
# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset.dropna(axis=1, thresh=2, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset[np.isfinite(dataset).all(1)]
# test_dataset = dataset.sample(frac=0.3, random_state=seed)
# dataset = dataset.reset_index()
array = dataset.values
X = array[:,:3064]
# X = array_OG[:,:3064]
Y = array[:,3064]
# Y = array_OG[:,3064]
Y = Y.astype('int32')
# Y = Y.astype('int32')
validation_size = 0.50
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size)
X_t = array_train[:,:3064]
Y_t = array_train[:,3064]
X_train = np.concatenate((X_train, X_t))
Y_train = np.concatenate((Y_train, Y_t))

X_all = np.concatenate((X, X_t))
Y_all = np.concatenate((Y, Y_t))

# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# print(Y_test)
# train = []
# for i in range(len(X_train)):
# 	row = X_train[i]
# 	row = np.append(X_train[i], Y_train[i])
# 	train.append(row)

# use_data = pd.DataFrame(data=train, dtype=np.float64)
# # print(use_data.shape)
# dataset_maj = use_data[use_data[3064] == 1]
# print(dataset_maj.shape)
# dataset_min = use_data[use_data[3064] == 0]
# print(dataset_min.shape)
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
# print(Y_validation)


# CLASSIFICATION METHODS
models = []
models.append(('GLM', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('BY', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('BAG', BaggingClassifier()))
models.append(('NNet', MLPClassifier()))
models.append(('RF', RandomForestClassifier()))
#models.append(('MARS', Earth()))
#models.append(('PLSR', PLSRegression()))
models.append(('BST', AdaBoostClassifier()))

num_fea = 50

# FEATURE SELECTION METHODS
sel = []

# sel.append(('CHSQ', SelectKBest(chi2, k=num_fea)))
# sel.append(('ANOVA', SelectKBest(f_classif, k=num_fea)))
# sel.append(('TSCR', SelectKBest(t_score.t_score, k=num_fea)))
# #sel.append(('GINI', SelectKBest(gini_index.gini_index, k=5)))
# sel.append(('FSCR', SelectKBest(fisher_score.fisher_score, k=num_fea)))
# sel.append(('RELF', SelectKBest(reliefF.reliefF, k=num_fea)))


output = open("disr50_mal.txt", "w")

# ACCURACY IS A MEASURE OF PREDICTIVE PERFORMANCE
scoring = 'roc_auc'

# 10-FOLD CROSS-VALIDATION OF EACH COMBINATION OF SELECTION/CLASSIFICATION

for name, model in models:
	for kind, selection in sel:
		print(name + ", " + kind)
		pipe = make_pipeline(MinMaxScaler(), selection, model)
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		cv_results = model_selection.cross_val_score(pipe, X_train, Y_train, cv=kfold, error_score='raise')
		msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		#output.write(msg)

		pipe.fit(X_train, Y_train)
		predictions = pipe.predict(X_test)
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_test, predictions)) + "\n")

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

def takeSecond(elem):
    return elem[1]
def WLCX(data, target, n_selected_features):
	pval = []
	for num in range(len(data[1])):
		x = data[:,num]
		pval.append([num, wilcoxon(x,target)[1]])
	pval.sort(key=takeSecond)
	idx = []
	for i in range(n_selected_features):
		idx.append(pval[i][0])
	return idx

MV_sel = []
# MV_sel.append(('WLCX', WLCX(X_all, Y_all, n_selected_features=num_fea)))
# print('WLCX')
# MV_sel.append(('MIM', MIM.mim(X, Y, n_selected_features=num_fea)))
# MV_sel.append(('MIFS', MIFS.mifs(X_all, Y_all, n_selected_features=num_fea)))
# print('MIFS')
# MV_sel.append(('MRMR', MRMR.mrmr(X_all, Y_all, n_selected_features=num_fea)))
# print('MRMR')
# MV_sel.append(('CIFE', CIFE.cife(X_all, Y_all, n_selected_features=num_fea)))
# print('CIFE')
# MV_sel.append(('JMI', JMI.jmi(X_all, Y_all, n_selected_features=num_fea)))
# print('JMI')
# MV_sel.append(('CMIM', CMIM.cmim(X_all, Y_all, n_selected_features=num_fea)))
# print('CMIM')
# MV_sel.append(('ICAP', ICAP.icap(X_all, Y_all, n_selected_features=num_fea)))
# print('ICAP')
MV_sel.append(('DISR', DISR.disr(X_all, Y_all, n_selected_features=num_fea)))
print('AFTER')

for name, model in models:
	for kind, idx in MV_sel:
		X_sel = X[:, idx[0:num_fea]]
		X_t_ = X_t[:, idx[0:num_fea]]
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=validation_size)
		# X_train = np.concatenate((X_train, X_t_))
		# Y_train = np.concatenate((Y_train, Y_t))
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
		msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		# print(msg)

		model.fit(X_train, Y_train)
		predictions = model.predict(X_validation)
		print(Y_validation)
		print(predictions)
		# predictions = model.predict(X_validation)
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + "\n")



# CALCULATING RSD FOR CLASSIFIERS
# idx = WLCX(X, Y, num_fea)
# X_sel = X[:, idx[0:num_fea]]
# X_t_ = X_t[:, idx[0:num_fea]]
# X_test = X_test[:,idx[0:num_fea]]
# for name, model in models:
# 	rsd = []
# 	for i in range(100):
# 		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=0.5)
# 		# X_train = np.concatenate((X_train, X_t_))
# 		# Y_train = np.concatenate((Y_train, Y_t))
# 		print(len(X_train))
# 		print(len(Y_train))
# 		model.fit(X_train, Y_train)
# 		predictions = model.predict(X_test)
# 		rsd.append(roc_auc_score(Y_test, predictions))
# 	output.write("Name: " + name + ", " + repr((np.std(rsd)/np.mean(rsd))*100) + "\n")
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X_t = scaler.fit_transform(X_t)
# print('hi')
stab_sel = []
# stab_sel.append(('CHSQ', chi2))
# stab_sel.append(('ANOVA', f_classif))
# stab_sel.append(('TSCR', t_score.t_score))
# #stab_sel.append(('GINI', gini_index.gini_index))
# stab_sel.append(('FSCR', fisher_score.fisher_score))
# stab_sel.append(('RELF', reliefF.reliefF))

def makeDFBinaryMatrix(selection):
	matrix = np.zeros((100, len(X[0])), dtype=int)
	for i in range(100):
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2)
		X_train = np.concatenate((X_train, X_t))
		Y_train = np.concatenate((Y_train, Y_t))
		best = SelectKBest(selection, k=num_fea)
		best.fit(X_train, Y_train)
		idx = best.get_support(indices=True)
		print(len(idx))
		for r in range(len(idx)):
			matrix[i,idx[r]] = 1
	return matrix

for kind, selection in stab_sel:
	print(kind)
	matrix = makeDFBinaryMatrix(selection)
	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")


s_sel = []
# s_sel.append(('WLCX', WLCX))
# # MV_sel.append(('MIM', MIM.mim(X, Y, n_selected_features=num_fea)))
# s_sel.append(('MIFS', MIFS.mifs))
# s_sel.append(('MRMR', MRMR.mrmr))
# s_sel.append(('CIFE', CIFE.cife))
# s_sel.append(('JMI', JMI.jmi))
# s_sel.append(('CMIM', CMIM.cmim))
# s_sel.append(('ICAP', ICAP.icap))
# s_sel.append(('DISR', DISR.disr))

def makeBinaryMatrix(selection):
	matrix = np.zeros((100, len(X[0])), dtype=int)
	for i in range(100):
		print(i)
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2)
		idx = selection(X_train,Y_train,n_selected_features=num_fea)
		for r in range(len(idx)):
			print(idx[r])
			matrix[i,idx[r]] = 1
	return matrix


for kind, selection in s_sel:
	print(kind)
	matrix = makeBinaryMatrix(selection)
	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")
