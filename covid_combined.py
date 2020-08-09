import csv
import pandas as pd
import numpy as np
from sklearn import model_selection
from tpot import TPOTClassifier
import tools
import matplotlib.pyplot as plt

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

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.impute._knn import KNNImputer

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

file2 = "clinical_features_SEL.csv"
f = open(file2)
csv_f = csv.reader(f)
dataset = pd.read_csv(file2, low_memory=False)
dataset["Grade-binary"] = pd.to_numeric(dataset["Grade-binary"], errors='coerce')
arrayClin = dataset.values
X = arrayClin[:,1:26]
Y = arrayClin[:,26]

# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = IterativeImputer(random_state=0)
# # imp = KNNImputer(n_neighbors=5)

train = "train.csv"
tr = open(train)
csv_tr = csv.reader(tr)
tr_data = pd.read_csv(train, usecols=range(0,1))
train = []
for line in csv_tr:
	Patient_DF = dataset.loc[dataset['PatientID']==line[0]]
	for index, row in Patient_DF.iterrows():
		train.append(row)
train = np.array(train)
X_train = train[:,1:26]
X_train = imp.fit_transform(X_train)
X_train = X_train.astype('float64')
Y_train = train[:,26]
Y_train = Y_train.astype('int32')
print(len(Y_train))


validation = "validation.csv"
val = open(validation)
csv_val = csv.reader(val)
val_data = pd.read_csv(validation, usecols=range(0,1))
validation = []
for line in csv_val:
	Patient_DF = dataset.loc[dataset['PatientID']==line[0]]
	for index, row in Patient_DF.iterrows():
		validation.append(row)
validation = np.array(validation)
X_validation = validation[:,1:26]
X_validation = imp.fit_transform(X_validation)
X_validation = X_validation.astype('float64')
Y_validation = validation[:,26]
Y_validation = Y_validation.astype('int32')
print(len(Y_validation))

test = "test.csv"
t = open(test)
csv_test = csv.reader(t)
test_data = pd.read_csv(test, usecols=range(0,1))
test = []
for line in csv_test:
	Patient_DF = dataset.loc[dataset['PatientID']==line[0]]
	for index, row in Patient_DF.iterrows():
		test.append(row)
test = np.array(test)
X_test = test[:,1:26]
X_test = imp.fit_transform(X_test)
X_test = X_test.astype('float64')
Y_test = test[:,26]
Y_test = Y_test.astype('int32')
print(len(Y_test))

foo = input()


# validation_size = 0.4
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

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
models.append(('BST', AdaBoostClassifier()))

num_fea = 20

# FEATURE SELECTION METHODS
sel = []

# sel.append(('CHSQ', SelectKBest(chi2, k=num_fea)))
# sel.append(('ANOVA', SelectKBest(f_classif, k=num_fea)))
# sel.append(('TSCR', SelectKBest(t_score.t_score, k=num_fea)))
# sel.append(('FSCR', SelectKBest(fisher_score.fisher_score, k=num_fea)))
# sel.append(('RELF', SelectKBest(reliefF.reliefF, k=num_fea)))

features_dict = dict()
output = open("scores.txt", "w")

scoring = 'roc_auc'

# UNIVARIATE FEATURE SELECTION X CLASSIFICATION (10 fold CV)

for name, model in models:
	for kind, selection in sel:
		pipe = make_pipeline(MinMaxScaler(), selection, model)
		pipe.fit(X_train, Y_train)
		
		# feat = pipe.named_steps['selectkbest'].get_support()
		# featNum = 0
		# for val in feat:
		# 	if(clinical[featNum] not in features_dict.keys()):
		# 		features_dict[clinical[featNum]] = 0
		# 	if val == True:
		# 		features_dict[clinical[featNum]]+=1
		# 	featNum+=1
	
		# predictions = pipe.predict(X_validation)
		# tn, fp, fn, tp = confusion_matrix(Y_validation, predictions).ravel()
		# output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + 
		# 	", Accuracy: " + repr(accuracy_score(Y_validation, predictions)) + ", Sensitivity: " + 
		# 	repr(recall_score(Y_validation, predictions)) + ", Specificity: " + repr(tn / (tn + fp)) + "\n")
		predictions = pipe.predict(X_test)
		tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_test, predictions)) + 
			", Accuracy: " + repr(accuracy_score(Y_test, predictions)) + ", Sensitivity: " + 
			repr(recall_score(Y_test, predictions)) + ", Specificity: " + repr(tn / (tn + fp)) + "\n")


for key in features_dict.keys():
	if(features_dict[key] > 15):
		print(key + '\n')


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
# MV_sel.append(('WLCX', WLCX(X, Y, n_selected_features=num_fea)))
# print('WLCX')
# MV_sel.append(('MIFS', MIFS.mifs(X, Y, n_selected_features=num_fea)))
# print('MIFS')
# MV_sel.append(('MRMR', MRMR.mrmr(X, Y, n_selected_features=num_fea)))
# print('MRMR')
# MV_sel.append(('CIFE', CIFE.cife(X, Y, n_selected_features=num_fea)))
# print('CIFE')
# MV_sel.append(('JMI', JMI.jmi(X, Y, n_selected_features=num_fea)))
# print('JMI')
# MV_sel.append(('CMIM', CMIM.cmim(X, Y, n_selected_features=num_fea)))
# print('CMIM')
# MV_sel.append(('ICAP', ICAP.icap(X, Y, n_selected_features=num_fea)))
# print('ICAP')
# MV_sel.append(('DISR', DISR.disr(X, Y, n_selected_features=num_fea)))
for name, model in models:
	for kind, idx in MV_sel:
		# X_sel = X[:, idx[0:num_fea]]
		# X_test_ = X_test[:,idx[0:num_fea]]
		X_train_ = X_train[:, idx[0:num_fea]]
		# X_validation_ = X_validation[:, idx[0:num_fea]]
		# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=validation_size, random_state=seed)
		# kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		# cv_results = model_selection.cross_val_score(model, X_train_, Y_train, cv=kfold)
		# msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		# output.write(msg)

		model.fit(X_train_, Y_train)
		# predictions = model.predict(X_validation_)
		# output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + "\n")
		# tn, fp, fn, tp = confusion_matrix(Y_validation, predictions).ravel()
		# output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + 
		# 	", Accuracy: " + repr(accuracy_score(Y_validation, predictions)) + ", Sensitivity: " + 
		# 	repr(recall_score(Y_validation, predictions)) + ", Specificity: " + repr(tn / (tn + fp)) + "\n")
		X_test_ = X_test[:, idx[0:num_fea]]
		predictions = model.predict(X_test_)
		tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_test, predictions)) + 
			", Accuracy: " + repr(accuracy_score(Y_test, predictions)) + ", Sensitivity: " + 
			repr(recall_score(Y_test, predictions)) + ", Specificity: " + repr(tn / (tn + fp)) + "\n")



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
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_train, Y_train, test_size=0.2)
		idx = selection(X_train,Y_train,n_selected_features=num_fea)
		for r in range(len(idx)):
			matrix[i,idx[r]] = 1
	return matrix


for kind, selection in s_sel:
	matrix = makeBinaryMatrix(selection)
	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")


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


# pipeline_optimizer = TPOTClassifier(generations=1, population_size=5, cv=5, verbosity=2, scoring='roc_auc')
# pipeline_optimizer.fit(X_train, Y_train)
# Y_pred = pipeline_optimizer.predict(X_validation)
# output.write("Accuracy: " + repr(accuracy_score(Y_validation, Y_pred)))
# output.write("Average Precision Score: " + repr(average_precision_score(Y_validation, Y_pred)))
# output.write("Kappa: " + repr(cohen_kappa_score(Y_validation, Y_pred)))
# output.write("Hamming Loss: " + repr(hamming_loss(Y_validation, Y_pred)))
# output.write("AUC: " + repr(roc_auc_score(Y_validation, Y_pred)))
# output.write("Sensitivity: " + repr(recall_score(Y_validation, Y_pred)))
# tn, fp, fn, tp = confusion_matrix(Y_validation, Y_pred).ravel()
# output.write("Specificity: " + repr(tn / (tn + fp)))
# pipeline_optimizer.export('tpot_exported_pipeline_COVID.py')



