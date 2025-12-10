from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

"""
This is is a classification problem. In a broad context, 
I am trying to identify spending patterns. I want to identify 
groups of high spending and predict

Problem statement:

What is my input: 

What is my output:

Identify the best train/test split for the data: 60/40, 70/30, 80/20
Use accuracy score and decision boundary plots to determine best fit

input:[[mean spending, variance, prob of class n]]

labels: a vector a labels corresponding to the input data

output: 

Data processing:
Manually editing csv data to include valid columns and combine redundant data entries

"""

#interpret and process the data
# RAW_DATA = pd.read_csv("credit_transactions.csv")
# RAW_DATA = pd.read_csv("Credit_Transactions_Cleaned.csv")
RAW_DATA = pd.read_csv("Credit_Transactions_Cleaned_v2.csv")
TRANSACT_DATA = RAW_DATA[RAW_DATA["Amount"] <= 0].copy()
TRANSACT_DATA["Amount"] = TRANSACT_DATA["Amount"] * -1

categorizedTransacts = dict()
categorizedDates = dict()
for i in range(len(TRANSACT_DATA["Amount"])):
    amount = TRANSACT_DATA["Amount"][i]
    date = TRANSACT_DATA["Effective Date"][i]
    description = TRANSACT_DATA["Extended Description"][i]
    if description not in categorizedTransacts.keys():
        categorizedTransacts[description], categorizedDates[description] = list(), list()
        categorizedTransacts[description].append(amount)
        categorizedDates[description].append(date)

    else:
        categorizedTransacts[description].append(amount)
        categorizedDates[description].append(date)

for i, j in categorizedDates.items():
    categorizedDates[i] = [datetime.strptime(d, "%m/%d/%Y") for d in j]

groupedPairs = dict() 
datesByWeek = dict()
transacByWeek = dict()

for category in categorizedDates.keys():
    categDates = categorizedDates[category]
    categTransac = categorizedTransacts[category]

    weeklyPairs = defaultdict(list)
    weeklyDates = defaultdict(list)
    weeklyTransac = defaultdict(list)

    for date, transac in zip(categDates, categTransac):
        y, w, _ = date.isocalendar()
        key = (y, w)

        weeklyPairs[key].append((date, transac))

        weeklyDates[key].append(date)
        weeklyTransac[key].append(transac)

    groupedPairs[category] = dict(weeklyPairs)
    datesByWeek[category] = dict(weeklyDates)
    transacByWeek[category] = dict(weeklyTransac)

weeklyMeans = dict()

for category, weeks in transacByWeek.items():
    weeklyMeans[category] = {}
    for key, tx_list in weeks.items():
        weeklyMeans[category][key] = [np.mean(tx_list), np.var(tx_list)]


inputData = []
for i, j in weeklyMeans.items():
    temp = []
    for k, v in j.items():
        # print(f"{k}, {v}, {i}")
        inputData.append([float(v[0]), float(v[1]), i])

#frequency based decision rule
categoryFreq = dict()
for i in TRANSACT_DATA["Extended Description"]:
    if i in categoryFreq.keys():
        categoryFreq[i]+=1

    else:
        categoryFreq[i] = 1

totalPurchases = sum(categoryFreq.values())
classRules = dict()
for i in categoryFreq.keys():
    classRules[i] = (categoryFreq[i])/totalPurchases

# for i, j in categoryFreq.items():
#     print(f"{i}: {j}")


# amount based decision rule
#make the data continuous by its date so that it can be given class labels
CONTIN_DATA = TRANSACT_DATA.copy()
CONTIN_DATA["Effective Date"] = pd.to_datetime(CONTIN_DATA["Effective Date"])
CONTIN_DATA['week'] = CONTIN_DATA["Effective Date"].dt.to_period('W')
weeklyTransactions = CONTIN_DATA.groupby('week')['Amount'].apply(list).to_dict()
normalizedTransactions = dict()
for i, j in weeklyTransactions.items():
    normalizedTransactions[i] = [np.mean(j), np.var(j)]

def padData(data):
    maxLen = 0
    for i in data:
        if maxLen < len(i):
            maxLen = len(i)
   
    for days in data:
        temp = days
        avg = np.average(days)
        if len(days) < maxLen:
            numElemsLeft = maxLen - len(days)
            for i in range(numElemsLeft):
                temp.append(avg)
            days = temp

    return data


def plotKMeans(k, init, iter, rand):
    model = KMeans(n_clusters=k, n_init=init, max_iter=iter, random_state=rand)
    model.fit()
    predictor = model.predict()

    for i in range(predictor.shape[0]):
        plt.scatter(x=predictor[i, 0], y = predictor[i, 1], marker = "x", color = "r")
      
    plt.title("K Means Clustering")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()
    return


def plotSVM(data):
    inputData, labels = data[:, 0:2], data[:, 2]

    #computing using linear kernal
    linearModel = svm.SVC(kernel = "linear")
    linearModel.fit(inputData, labels)
    linPredict = linearModel.predict(data[:, 0:2])

    for i in range(linPredict.shape[0]):
        if linPredict[i] == 1:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "x", color = "r")
        
        elif linPredict[i] == 2:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "o", color = "g")

        else:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "*", color = "b")
      
    plt.title("SVM with linear kernal")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()  

    #computing using radial basis function kernal
    rbfModel = svm.SVC(kernel = "rbf")
    rbfModel.fit(inputData, labels)
    rbfPredict = rbfModel.predict(data[:, 0:2])

    for i in range(rbfPredict.shape[0]):
        if rbfPredict[i] == 1:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "x", color = "r")
        
        elif rbfPredict[i] == 2:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "o", color = "g")

        else:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "*", color = "b")
    
    plt.title("SVM with Radial Kernal")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()
    return

def plotGMM(data):
    return


def plotLogReg(data):
    inputData, labels = data[:, 0:2], data[:, 2]
    logModel = LogisticRegression()
    logModel.fit(inputData, labels)
    logPredict = logModel.predict(data[:, 0:2])
    
    #plots the original data points with their actual classifications

    #adds all the data using the different predicted class associations for each point to the plot
    for i in range(logPredict.shape[0]):
        if logPredict[i] == 1:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "x", color = "r")
        
        elif logPredict[i] == 2:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "o", color = "g")

        elif logPredict[i] == 3:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "*", color = "b")

        else:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "*", color = "m")
    
    #adds the decision boundary to the plots
    # plotDescBound(logModel)

    plt.title("Plotting data using Log regression")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()
    
    #calculating cross-entropy loss and number of points missclassified
    yPred = logModel.predict_proba(data[:, 0:2])
    ceLoss = 0
    
    for i in range(yPred.shape[0]):
        ceLoss += np.log(yPred[i,int(labels[i]-1)])
    
    ceLoss *= -1
    classAcc = logModel.score(data[:, 0:2],labels)

    return ceLoss, (1 - classAcc)

#plots the decision boundary using data
def plotDescBound(model):
    [x, y] = np.meshgrid(np.linspace(-2, 3, num = 100), np.linspace(-1.5, 1.5, num=100))
    for i in tqdm(range(x.shape[0])):
        for j in range(y.shape[0]):
            p = np.array([[x[i, j], y[i, j]]])
            ptClass = model.predict(p)
            
            if ptClass == 1:
                plt.scatter(x=x[i,j], y=y[i,j], color = "r", alpha = 0.1, marker = "o")
            
            elif ptClass == 2:
                plt.scatter(x=x[i,j], y=y[i,j], color = "g", alpha = 0.1, marker = "o")
            
            elif ptClass == 3:
                plt.scatter(x=x[i,j], y=y[i,j], color = "b", alpha = 0.1, marker = "o")
    return

def plotDecisionTree(data):
    return


def plotRandForest(data):
    return


if __name__ == "__main__":
    pass
    data = np.array(inputData)
    plotSVM(data)
    # plotLogReg(data)
    
