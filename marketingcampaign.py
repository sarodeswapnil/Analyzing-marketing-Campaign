import json, operator
import sys
from collections import OrderedDict
import pprint
from pprint import pprint
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import random
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def sample_data(indep_var, dep_var):
	training_data_X=[]
	training_data_Y=[]
	test_data_X=[]
	test_data_Y=[]
	
	for i in range(len(indep_var)):		
		temp_x=[]
		#select every 5th element(20%) to the test data
		if i%5 == 0:
			for j in range(len(indep_var[i])):
				temp_x.append(float(indep_var[i][j]))
			test_data_X.append(temp_x)
			test_data_Y.append(dep_var[i])
		else:
		#training data will make up 80% of data
			for j in range(len(indep_var[i])):
				temp_x.append(float(indep_var[i][j]))
			training_data_X.append(temp_x)
			training_data_Y.append(dep_var[i])
	return training_data_X,training_data_Y, test_data_X,test_data_Y

def oversample_data(indep_var, dep_var):
	training_data_X=[]
	training_data_Y=[]
	test_data_X=[]
	test_data_Y=[]
	resp_ctr = 0
	for i in range(len(indep_var)):		
		temp_x=[]
		if dep_var[i]==1:
			#select all responds which form 6.31% of all data
			if resp_ctr%4==0:
				#every 4th element (25%) of the data goes to test set
				for j in range(len(indep_var[i])):
					temp_x.append(float(indep_var[i][j]))
				test_data_X.append(temp_x)
				test_data_Y.append(dep_var[i])
			else:
				#rest of the elements (75%) of the data goes to training set
				for j in range(len(indep_var[i])):
					temp_x.append(float(indep_var[i][j]))
				training_data_X.append(temp_x)
				training_data_Y.append(dep_var[i])

			resp_ctr+=1
	non_resp_ctr = 0
	while non_resp_ctr <= resp_ctr:		
	#loop ends when number of non-responds is equal to number of responds
		temp_x=[]
		#randomly select element from data
		ptr = random.randint(0,len(indep_var)-1)
		if dep_var[ptr]==0:
			if non_resp_ctr%4 == 0:
				#every 4th element (25%) of the data goes to test set
				for j in range(len(indep_var[ptr])):
					temp_x.append(float(indep_var[ptr][j]))
				test_data_X.append(temp_x)
				test_data_Y.append(dep_var[ptr])
			else:
				#rest of the elements (75%) of the data goes to training set
				for j in range(len(indep_var[i])):
					temp_x.append(float(indep_var[ptr][j]))
				training_data_X.append(temp_x)
				training_data_Y.append(dep_var[ptr])
			non_resp_ctr+=1

	return training_data_X,training_data_Y, test_data_X,test_data_Y
 

def plot_roc(test_data_Y, predicted_Y, caption):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(len(test_data_Y)):
		#calculate the tpr and fpr to plot the roc curve
		fpr[i], tpr[i], _ = roc_curve(test_data_Y, predicted_Y)
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(test_data_Y, predicted_Y)
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='red',
		 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Q j. ROC for '+caption)
	plt.legend(loc="lower right")
	plt.show()

	Prob_Y = clf.predict_proba(test_data_X)
	skplt.metrics.plot_lift_curve(test_data_Y, Prob_Y, title="Lift Curve for "+caption)
	plt.show()	


def fit_classifier(clf, training_data_X, training_data_Y, test_data_X):
	#fit the classifier to the data
	clf.fit(training_data_X, training_data_Y)
	predicted_Y=[]
	for i in range(len(test_data_X)):	
		#calculated predicted Y from the classifier
		predicted_Y.append(clf.predict(test_data_X[i]))
	return predicted_Y

def print_alpha_data(data):
	#sort the data by 'NAME'
	alpha_data = sorted(data, key =lambda k:k['NAME'])
	print "Q h. Printing alphabetized list by name 1-10:"	
	for i in range(10):
		pprint(alpha_data[i])
	print "Q h. Printing alphabetized list by name 20000-20010:"	
	for i in range(20000,20011):
		pprint(alpha_data[i])
	#sort the data by the last word in 'NAME'
	alpha_data = sorted(data, key =lambda k:str.split(str(k['NAME']))[-1])
	raw_input("Press key to continue...")
	print "Q i. Printing alphabetized list by last name 1-10:"	
	for i in range(10):
		pprint(alpha_data[i])
	print "Q i. Printing alphabetized list by last name 20000-20010:"	
	for i in range(20000,20011):
		pprint(alpha_data[i])
	raw_input("Press key to continue...")


def print_dendro(indep_var):
	dendro_data=[]
	for i in range(len(indep_var)):		
		for j in range(len(indep_var[i])-1):
			indep_var[i][j]=float(indep_var[i][j])
		if i%10 == 0:
			dendro_data.append(indep_var[i])
	Z = linkage(np.array(dendro_data), method='ward', metric='euclidean') 
	plt.title('Q f. Hierarchical Clustering Dendrogram using all features and 1/10th observations')
	plt.xlabel('Targeted Customers')                                                        
	plt.ylabel('distance')
	dendrogram(
	    Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=8.,  # font size for the x axis labels
	)
	#display both figures
	plt.axhline(y=16,color="blue")
	plt.show()

def print_histo(data):

	respond = []
	non_respond = []
	for i in range(len(data)):
		#create separate lists of wealth index for responders and non-responders
		if dep_var[i]==1:
			respond.append(float(data[i]['WEALTH_INDEX']))
		else:
			non_respond.append(float(data[i]['WEALTH_INDEX']))

	plt.title("Q g. Histogram of WEALTH_INDEX for the responders and non-responders by counts")
	labels=["Responders", "Non_responders"]
	bins=np.linspace(0,50,20)
	#create histogram with v-stacked bars with non-normalized data
	plt.hist([respond,non_respond], bins, histtype='bar',normed=False, alpha=0.8)

	plt.legend(labels)
	plt.show()

	plt.title("Q g. Histogram of WEALTH_INDEX for the responders and non-responders by percent")
	plt.legend(labels)
	#create histogram with v-stacked bars with normalized data
	plt.hist([respond,non_respond],bins, histtype='bar',normed=True, alpha=0.8)
	plt.show()

def normalize_data(indep_var):
	#Min-max scaler is used to normalize numerical data to a scale of 0 to 1
	min_max_scaler = preprocessing.MinMaxScaler()
	indep_var = np.array(indep_var)
	for i in [0,2,3,5,6,7,8]:
		indep_var[:,i]=	min_max_scaler.fit_transform(indep_var[:,i])
	for i in [1,4]:
		#Label encoder is used encode categorical data
		le = preprocessing.LabelEncoder()
		le.fit(indep_var[:,i])
		indep_var[:,i]=le.transform(indep_var[:,i])
	return indep_var

def extract_data(data):
	for record in data:
		temp = []
		#extract data from the data and convert to int or float for processing
		for key in key_attributes:
			if key == 'TARGET_B':
				dep_var.append(int(record['TARGET_B']))
			elif key == 'RFA_2F' or key == 'LASTGIFT'or key =='LASTDATE' \
				or key == 'INCOME' or key == 'FISTDATE':
				temp.append(int(record[key]))
			elif key == 'AVGGIFT' or key == 'WEALTH_INDEX':
				temp.append(float(record[key]))
			elif key == 'RFA_2A' or key == 'PEPSTRFL':
				temp.append(record[key])
		indep_var.append(temp)
	return dep_var, indep_var

#Open Json file
with open('assess2_data.json') as json_file:  
	data = json.load(json_file,object_pairs_hook=OrderedDict)
	print "Q a. Attributes/Fields in the file are:"
	key_attributes = [str(keys) for keys in data[0]]
	pprint (key_attributes)
	print "Q a. Number of records are:", len(data)
	raw_input("Press key to continue...")
	#Create dictionaries to segregate seen and unseen records
	seen = OrderedDict()
	duplicate = OrderedDict()
	rfa2f_seen = OrderedDict()
	rfa2a_seen = OrderedDict()
	wealth_seen = OrderedDict()

	#Check all records in the file

	for record in data:
		oname = record["NAME"]
		#Put all first-time records in seen
		if oname not in seen:
			seen[oname] = record
		#Put all duplicate records in duplicate
		else:
			duplicate[oname]=record

	#print duplicate names
	print "Q b. Following are the duplicate names"
	for record in duplicate:
		print record
	raw_input("Press key to continue...")

	#counter to identify index of rows with missing value
	ctr=0
	missing_ctr=0
	for record in data:
		for key in key_attributes:
			if record[key] == -9999 or record[key] == "-9999": 
				del(data[ctr])
				missing_ctr+=1
		ctr+=1
	print "Q c. Total number of  missing values :", missing_ctr
	raw_input("Press key to continue...")
	rfa_2f_ctr={}
	rfa_2a_ctr={}
	respond_ctr={'1':0,'0':0}
	#Put all unique values in seen
	for record in data:
		if record["RFA_2F"] not in rfa2f_seen:
			rfa2f_seen[record["RFA_2F"]] = record
			rfa_2f_ctr.update({str(record["RFA_2F"]):1})
		else:
			rfa_2f_ctr[str(record["RFA_2F"])]=rfa_2f_ctr[str(record["RFA_2F"])]+1			

		if record["RFA_2A"] not in rfa2a_seen:
			rfa2a_seen[record["RFA_2A"]] = record
			rfa_2a_ctr.update({str(record["RFA_2A"]):1})
		else:
			rfa_2a_ctr[str(record["RFA_2A"])]=rfa_2a_ctr[str(record["RFA_2A"])]+1			

		if record["WEALTH_INDEX"] not in wealth_seen:
			wealth_seen[record["WEALTH_INDEX"]] = record
		respond_ctr[str(record["TARGET_B"])]=respond_ctr[str(record["TARGET_B"])]+1			
		

	#print unique values and the frequency
	print "Q d. i. Following are the list of values for RFA_2F and their frequency"
	for record in rfa2f_seen:
		print record, ":", rfa_2f_ctr[str(record)]
	print "Q d. ii. Following are the list of values for RFA_2A and their frequency"
	for record in rfa2a_seen:
		print record, ":", rfa_2a_ctr[str(record)]
	raw_input("Press key to continue...")
	print "Q d. iii. Following are the list of values for Wealth_Index"
	for record in wealth_seen:
		pprint( record)
	print "Min in Wealth_Index", min(wealth_seen)
	print "Max in Wealth_Index", max(wealth_seen)
	raw_input("Press key to continue...")
	dep_var = []
	indep_var=[]
	print "Q e. Proportion of responders"
	print  (float(respond_ctr['1']) /float(respond_ctr['0']+respond_ctr['1']))*100,"%"
	raw_input("Press key to continue...")

	dep_var, indep_var = extract_data(data)
	#Normalize data
	indep_var = normalize_data(indep_var)
	print_dendro(indep_var)
	print_histo(data)
	print_alpha_data(data)
	
#	training_data_X,training_data_Y, test_data_X,test_data_Y=sample_data(indep_var, dep_var)
	training_data_X,training_data_Y, test_data_X,test_data_Y=oversample_data(indep_var, dep_var)
	#create the decision tree classifier with max depth of 10
	clf = tree.DecisionTreeClassifier()
	clf = tree.DecisionTreeClassifier(max_depth=10)
	predicted_Y=fit_classifier(clf, training_data_X, training_data_Y, test_data_X)
	plot_roc(test_data_Y,predicted_Y,"Decision Tree")

	#create the Neural network classifier
	clf = MLPClassifier()
	predicted_Y=fit_classifier(clf, training_data_X, training_data_Y, test_data_X)
	plot_roc(test_data_Y,predicted_Y, "Neural Network")

	#create the Logistic Regression classifier
	clf = linear_model.LogisticRegression()
	predicted_Y=fit_classifier(clf, training_data_X, training_data_Y, test_data_X)
	plot_roc(test_data_Y,predicted_Y, "Logistic Regression")

