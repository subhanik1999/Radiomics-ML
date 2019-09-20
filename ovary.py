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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
seed = 7
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

mapping = "ovary mapping.csv"
m = open(mapping)
csv_m = csv.reader(m)
titles = next(csv_m)
mdata = pd.read_csv(mapping, names=titles, skiprows=1)
mdata.dropna(inplace=True)
mapped = dict()
for line in csv_m:
	mapped[line[4]] = line[0]

file = "radiomics__image_ovary.csv"
f = open(file)
csv_f = csv.reader(f)
features = next(csv_f)
dataset = pd.read_csv(file, names=features, skiprows=1, low_memory=False)
dataset["Grade-binary"] = pd.to_numeric(dataset["Binary-grade"], errors='coerce')

# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset.dropna(axis=1, thresh=2, inplace=True)
dataset.dropna(inplace=True)
# print(dataset.dtypes)
array = dataset.values
X = array[:,1:3065]
Y = array[:,3065]
# print(dataset.shape)
# dataset_maj = dataset[dataset["Binary-grade"] == 0]
# print(dataset_maj.shape)
# dataset_min = dataset[dataset["Binary-grade"] == 1]
# print(dataset_min.shape)
# dataset_maj_downsamp = resample(dataset_maj, replace=True, n_samples=324, random_state=seed)
# dataset_downsamp = pd.concat([dataset_min, dataset_maj_downsamp])
# print(dataset_downsamp.shape)
# array = dataset_downsamp.values
# # X = array[:,1:3065]
# # Y = array[:,3065]

# dataset = dataset[np.isfinite(dataset).all(1)]
train = "train_ovary.csv"
tr = open(train)
csv_tr = csv.reader(tr)
titles = next(csv_tr)
tr_data = pd.read_csv(train, names=titles, usecols=range(0,1), skiprows=1)
# dataset = dataset.set_index('PatientID')
# print(len(mapped))


train = []
for line in csv_tr:
	ids = line[0]
	if ids in mapped:
		pid = mapped[ids]
		print(pid)
		Patient_DF = dataset.loc[dataset['PatientID']==pid]
		for index, row in Patient_DF.iterrows():
			train.append(row)

train = np.array(train)
foo=input()
validation = "validation_ovary.csv"
val = open(validation)
csv_val = csv.reader(val)
titles = next(csv_val)
val_data = pd.read_csv(validation, names=titles, usecols=range(0,1), skiprows=1)

validation = []
for line in csv_val:
	ids = line[0]
	if ids in mapped:
		pid = mapped[ids]
		Patient_DF = dataset.loc[dataset['PatientID']==pid]
		for index, row in Patient_DF.iterrows():
			validation.append(row)

validation = np.array(validation)


test = "test_ovary.csv"
t = open(test)
csv_test = csv.reader(t)
titles = next(csv_test)
test_data = pd.read_csv(test, names=titles, usecols=range(0,1), skiprows=1)

test = []
for line in csv_test:
	ids = line[0]
	if ids in mapped:
		pid = mapped[ids]
		Patient_DF = dataset.loc[dataset['PatientID']==pid]
		for index, row in Patient_DF.iterrows():
			test.append(row)

test = np.array(test)
X_train = train[:,1:3065]
Y_train = train[:, 3065]
Y_train = Y_train.astype('int32')
X_validation = validation[:,1:3065]
X_validation = X_validation.astype('float64')
Y_validation = validation[:, 3065]
Y_validation = Y_validation.astype('int32')
X_test = test[:,1:3065]
# X_test = X_test.astype('float64')
Y_test = test[:, 3065]
Y_test = Y_test.astype('int32')
# print(len(Y_test))

# array = dataset.values
# X = array[1:,1:3065]
# Y = array[1:,3065]
# print(Y)
# Y = Y.astype('int32')
# validation_size = 0.30
seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# # CLASSIFICATION METHODS
models = []
log = LogisticRegression()
penalty = ['l1', 'l2']
C = [0.001,0.01,0.1,1,10,100,1000]
# solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
max_iter=[100,110,120,130,140]
param_grid = dict(C=C, penalty=penalty, max_iter=max_iter)
grid = GridSearchCV(estimator=log, param_grid=param_grid, cv=10, n_jobs=-1)
# print('glm')
# models.append(('GLM', grid))
# models.append(('GLM', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('DT', DecisionTreeClassifier()))
# models.append(('BY', GaussianNB()))
# models.append(('SVM', SVC()))
# models.append(('BAG', BaggingClassifier()))
net = MLPClassifier()
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['identity', 'tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9],
    'learning_rate': ['constant', 'invscaling', 'adaptive']}
grid_ = GridSearchCV(estimator=net, param_grid=parameter_space, cv=10, n_jobs=-1)
# print('nnet')
models.append(('NNet', MLPClassifier()))
# models.append(('NNet', grid_))
# models.append(('RF', RandomForestClassifier()))
# # models.append(('MARS', Earth()))
# # models.append(('PLSR', PLSRegression()))
# models.append(('BST', AdaBoostClassifier()))
# ens = VotingClassifier(estimators=[('NNet', MLPClassifier()), ('GLM', LogisticRegression())], voting='soft')
# models.append(('ensemble', ens))
num_fea = 50

# # FEATURE SELECTION METHODS
sel = []

# sel.append(('CHSQ', SelectKBest(chi2, k=num_fea)))
# sel.append(('ANOVA', SelectKBest(f_classif, k=num_fea)))
# sel.append(('TSCR', SelectKBest(t_score.t_score, k=num_fea)))
# sel.append(('GINI', SelectKBest(gini_index.gini_index, k=5)))
sel.append(('FSCR', SelectKBest(fisher_score.fisher_score, k=num_fea)))
# sel.append(('RELF', SelectKBest(reliefF.reliefF, k=num_fea)))


output = open("namesFeat.txt", "w")

scoring = 'roc_auc'

# UNIVARIATE FEATURE SELECTION X CLASSIFICATION (10 fold CV)

for name, model in models:
	for kind, selection in sel:
		print(kind)
		pipe = make_pipeline(MinMaxScaler(), selection, model)
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		cv_results = model_selection.cross_val_score(pipe, X_train, Y_train, cv=kfold)
		# msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		#output.write(msg)

		pipe.fit(X_train, Y_train)
		feat = pipe.named_steps['selectkbest'].get_support()
		featNum = 0
		for val in feat:
			featNum+=1
			if val == True:
				output.write(repr(featNum) + '\n')
		print(featNum)
		predictions = pipe.predict(X_test)
		# print(predictions)
		# print(Y_test)
		# print(str(roc_auc_score(Y_test, predictions)))
		# print(test[:,0])
		# foo = input()
		# output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + "\n")
		# tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
		# output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_test, predictions)) + 
		# 	", Accuracy: " + repr(accuracy_score(Y_test, predictions)) + ", Sensitivity: " + 
		# 	repr(recall_score(Y_test, predictions)) + ", Specificity: " + repr(tn / (tn + fp)) + "\n")
		# print(name)
		# print(model.best_params_)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.fit_transform(X_validation)
X_test = scaler.fit_transform(X_test)

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

# # MULTIVARIATE FEATURE SELECTION X CLASSIFICATION (10 fold CV)

# # print('BEFORE')
MV_sel = []
# MV_sel.append(('WLCX', WLCX(X, Y, n_selected_features=num_fea)))
# print('WLCX')
# #MV_sel.append(('MIM', MIM.mim(X, Y, n_selected_features=num_fea)))
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
		print(name + ", " + kind)
		# X_sel = X[:, idx[0:num_fea]]
		X_train_ = X_train[:, idx[0:num_fea]]
		X_validation_ = X_validation[:, idx[0:num_fea]]
		# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=validation_size, random_state=seed)
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		
		cv_results = model_selection.cross_val_score(model, X_train_, Y_train, cv=kfold)
		msg = "%s %s: %f (%f)\n" % (kind, name, cv_results.mean(), cv_results.std())
		#output.write(msg)

		model.fit(X_train_, Y_train)
		predictions = model.predict(X_validation_)
		output.write("Name: " + name + ", " + "Sel: " + kind + "| " + "AUC: " + str(roc_auc_score(Y_validation, predictions)) + "\n")

# # CALCULATING RSD FOR CLASSIFIERS

# # idx = WLCX(X, Y, num_fea)
# # X_sel = X[:, idx[0:num_fea]]
# # for name, model in models:
# # 	rsd = []
# # 	for i in range(100):
# # 		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_sel, Y, test_size=0.5)
# # 		model.fit(X_train, Y_train)
# # 		predictions = model.predict(X_validation)
# # 		rsd.append(roc_auc_score(Y_validation, predictions))
# # 	output.write("Name: " + name + ", " + repr((np.std(rsd)/np.mean(rsd))*100) + "\n")

# # STABILITY OF CLASSIFICATION METHODS (MULTIVARIATE) ... takes a long time to run

# s_sel = []
# # s_sel.append(('WLCX', WLCX))
# # # s_sel.append(('MIM', MIM.mim(X, Y, n_selected_features=num_fea)))
# # s_sel.append(('MIFS', MIFS.mifs))
# # s_sel.append(('MRMR', MRMR.mrmr))
# # s_sel.append(('CIFE', CIFE.cife))
# # s_sel.append(('JMI', JMI.jmi))
# # s_sel.append(('CMIM', CMIM.cmim))
# # s_sel.append(('ICAP', ICAP.icap))
# # s_sel.append(('DISR', DISR.disr))

# def makeBinaryMatrix(selection):
# 	matrix = np.zeros((100, len(X[0])), dtype=int)
# 	for i in range(100):
# 		print(i)
# 		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2)
# 		idx = selection(X_train,Y_train,n_selected_features=num_fea)
# 		for r in range(len(idx)):
# 			matrix[i,idx[r]] = 1
# 	return matrix


# for kind, selection in s_sel:
# 	matrix = makeBinaryMatrix(selection)
# 	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")

# # STABILITY OF CLASSIFICATION METHODS (UNIVARIATE)

# stab_sel = []
# # stab_sel.append(('CHSQ', chi2))
# # stab_sel.append(('ANOVA', f_classif))
# # stab_sel.append(('TSCR', t_score.t_score))
# # #stab_sel.append(('GINI', gini_index.gini_index))
# # stab_sel.append(('FSCR', fisher_score.fisher_score))
# # stab_sel.append(('RELF', reliefF.reliefF))

# def makeDFBinaryMatrix(selection):
# 	matrix = np.zeros((100, len(X[0])), dtype=int)
# 	for i in range(100):
# 		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2)
# 		best = SelectKBest(selection, k=num_fea)
# 		best.fit(X_train, Y_train)
# 		idx = best.get_support(indices=True)
# 		for r in range(len(idx)):
# 			matrix[i,idx[r]] = 1
# 	return matrix

# for kind, selection in stab_sel:
# 	print(kind)
# 	matrix = makeDFBinaryMatrix(selection)
# 	output.write(kind + ": " + repr(tools.getStability(matrix))+"\n")


# pipeline_optimizer = TPOTClassifier(generations=10, population_size=10, cv=5, verbosity=2, scoring='roc_auc')
# pipeline_optimizer.fit(X_train, Y_train)
# Y_pred = pipeline_optimizer.predict(X_validation)
# print("Accuracy: " + repr(accuracy_score(Y_validation, Y_pred)))
# print("Average Precision Score: " + repr(average_precision_score(Y_validation, Y_pred)))
# print("Kappa: " + repr(cohen_kappa_score(Y_validation, Y_pred)))
# print("Hamming Loss: " + repr(hamming_loss(Y_validation, Y_pred)))
# print("AUC: " + repr(roc_auc_score(Y_validation, Y_pred)))
# print("Sensitivity: " + repr(recall_score(Y_validation, Y_pred)))
# tn, fp, fn, tp = confusion_matrix(Y_validation, Y_pred).ravel()
# print("Specificity: " + repr(tn / (tn + fp)))
# pipeline_optimizer.export('tpot_exported_pipeline_o.py')
