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
from sklearn.ensemble import GradientBoostingClassifier
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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

file = "radiomics__(tumor)__image_GRADE.csv"

f = open(file)
csv_f = csv.reader(f)
features = next(csv_f)
dataset = pd.read_csv(file, names=features, usecols=range(1,3089), skiprows=1, low_memory=False)
dataset["Grade-binary"] = pd.to_numeric(dataset["Grade-binary"], errors='coerce')
print(dataset.dtypes)
# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset.dropna(axis=1, thresh=2, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset[np.isfinite(dataset).all(1)]
print(dataset.shape)
array = dataset.values
X = array[1:,1:3064]
Y = array[1:,3064]
print(Y.shape)
Y = Y.astype('int32')
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

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
# models.append(('MARS', Earth()))
# models.append(('PLSR', PLSRegression()))
models.append(('BST', AdaBoostClassifier()))

num_fea = 10

# FEATURE SELECTION METHODS
sel = []

# sel.append(('CHSQ', SelectKBest(chi2, k=num_fea)))
# sel.append(('ANOVA', SelectKBest(f_classif, k=num_fea)))
# sel.append(('TSCR', SelectKBest(t_score.t_score, k=num_fea)))
# #sel.append(('GINI', SelectKBest(gini_index.gini_index, k=5)))
# sel.append(('FSCR', SelectKBest(fisher_score.fisher_score, k=num_fea)))
# sel.append(('RELF', SelectKBest(reliefF.reliefF, k=num_fea)))


output = open("mv10_g2.txt", "w")

scoring = 'roc_auc'

# UNIVARIATE FEATURE SELECTION X CLASSIFICATION (10 fold CV)

for name, model in models:
	for kind, selection in sel:
		print(kind)
		pipe = make_pipeline(MinMaxScaler(), selection, model)
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		cv_results = model_selection.cross_val_score(pipe, X_train, Y_train, cv=kfold)
		msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		#output.write(msg)

		pipe.fit(X_train, Y_train)
		predictions = pipe.predict(X_validation)
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + "\n")

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# WILCOXON SCORE FUNCTION

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

# MULTIVARIATE FEATURE SELECTION X CLASSIFICATION (10 fold CV)

# print('BEFORE')
MV_sel = []
MV_sel.append(('WLCX', WLCX(X, Y, n_selected_features=num_fea)))
print('WLCX')
##MV_sel.append(('MIM', MIM.mim(X, Y, n_selected_features=num_fea)))
MV_sel.append(('MIFS', MIFS.mifs(X, Y, n_selected_features=num_fea)))
print('MIFS')
MV_sel.append(('MRMR', MRMR.mrmr(X, Y, n_selected_features=num_fea)))
print('MRMR')
MV_sel.append(('CIFE', CIFE.cife(X, Y, n_selected_features=num_fea)))
print('CIFE')
MV_sel.append(('JMI', JMI.jmi(X, Y, n_selected_features=num_fea)))
print('JMI')
MV_sel.append(('CMIM', CMIM.cmim(X, Y, n_selected_features=num_fea)))
print('CMIM')
MV_sel.append(('ICAP', ICAP.icap(X, Y, n_selected_features=num_fea)))
print('ICAP')
MV_sel.append(('DISR', DISR.disr(X, Y, n_selected_features=num_fea)))
for name, model in models:
	for kind, idx in MV_sel:
		X_sel = X[:, idx[0:num_fea]]
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=validation_size, random_state=seed)
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
		msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		#output.write(msg)

		model.fit(X_train, Y_train)
		predictions = model.predict(X_validation)
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + "\n")

# CALCULATING RSD FOR CLASSIFIERS

# idx = WLCX(X, Y, num_fea)
# X_sel = X[:, idx[0:num_fea]]
# for name, model in models:
# 	rsd = []
# 	for i in range(100):
# 		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=0.5)
# 		model.fit(X_train, Y_train)
# 		predictions = model.predict(X_validation)
# 		rsd.append(roc_auc_score(Y_validation, predictions))
# 	output.write("Name: " + name + ", " + repr((np.std(rsd)/np.mean(rsd))*100) + "\n")

# STABILITY OF CLASSIFICATION METHODS (MULTIVARIATE) ... takes a long time to run

s_sel = []
# s_sel.append(('WLCX', WLCX))
# # s_sel.append(('MIM', MIM.mim(X, Y, n_selected_features=num_fea)))
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
			matrix[i,idx[r]] = 1
	return matrix


for kind, selection in s_sel:
	matrix = makeBinaryMatrix(selection)
	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")

# STABILITY OF CLASSIFICATION METHODS (UNIVARIATE)

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
		best = SelectKBest(selection, k=num_fea)
		best.fit(X_train, Y_train)
		idx = best.get_support(indices=True)
		for r in range(len(idx)):
			matrix[i,idx[r]] = 1
	return matrix

for kind, selection in stab_sel:
	print(kind)
	matrix = makeDFBinaryMatrix(selection)
	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")




