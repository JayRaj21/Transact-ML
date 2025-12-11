from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# importing data from CSV file in the same directory
RAW_DATA = pd.read_csv("Credit_Transactions_Cleaned_v2.csv")
SPENDING_DATA = RAW_DATA[RAW_DATA["Amount"] <= 0].copy()
SPENDING_DATA["Amount"] = SPENDING_DATA["Amount"] * -1

# separates and categorizes spending by weekly and description
categorizedTransacts = dict()
categorizedDates = dict()
for i in range(len(SPENDING_DATA["Amount"])):
    amount = SPENDING_DATA["Amount"][i]
    date = SPENDING_DATA["Effective Date"][i]
    description = SPENDING_DATA["Extended Description"][i]
    if description not in categorizedTransacts.keys():
        categorizedTransacts[description], categorizedDates[description] = list(), list()
        categorizedTransacts[description].append(amount)
        categorizedDates[description].append(date)

    else:
        categorizedTransacts[description].append(amount)
        categorizedDates[description].append(date)

for i, j in categorizedDates.items():
    categorizedDates[i] = [datetime.strptime(d, "%m/%d/%Y") for d in j]

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

    transacByWeek[category] = dict(weeklyTransac)

#calculating the mean and variance spending in a weekly interval
weeklyMeansVar = dict()
for category, weeks in transacByWeek.items():
    weeklyMeansVar[category] = {}
    for key, tx_list in weeks.items():
        weeklyMeansVar[category][key] = [np.mean(tx_list), np.var(tx_list)]

# aggregates data in format that can be accepted by scikit-learn algos
processedData = []
for i, j in weeklyMeansVar.items():
    temp = []
    for k, v in j.items():
        processedData.append([float(v[0]), float(v[1]), i])

#frequency based decision rule
categoryFreq, classLabels, count = dict(), dict(), 0
for i in SPENDING_DATA["Extended Description"]:
    if i in categoryFreq.keys():
        categoryFreq[i]+=1

    else:
        categoryFreq[i] = 1
        classLabels[i] = count
        count+=1

totalPurchases, classRules = sum(categoryFreq.values()), dict()
for i in categoryFreq.keys():
    classRules[i] = (categoryFreq[i])/totalPurchases

# function for implementing a k means clustering
def plotKMeans(xTrain, yTrain, xTest, yTest, k):
    model = KMeans(n_clusters=k)
    model.fit(xTrain, yTrain)
    predictor = model.fit_predict(xTrain, yTrain)

    # for i in tqdm(range(predictor.shape[0])):
    #     # plt.scatter(x=predictor[i, 0], y = predictor[i, 1], marker = "x", color = "r")
    #     print(predictor[i])
    # plt.title("K Means Clustering")
    # plt.xlabel("x data")
    # plt.ylabel("y data")
    # plt.show()

    print(f"SSE = {-(model.score(xTrain, yTrain))}")
    print(f"SSE = {-(model.score(xTest, yTest))}")
    # print(model.score(xTrain, yTrain))
    return

# function for implementing SVM classifier
def plotSVM(xTrain, yTrain, xTest, yTest):
    #computing using linear kernal
    linearModel = svm.SVC(kernel = "linear")
    linearModel.fit(xTrain, yTrain)
    linPredict = linearModel.predict(xTrain)

    # for i in range(linPredict.shape[0]):
    #     if linPredict[i] == 1:
    #         plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "x", color = "r")
        
    #     elif linPredict[i] == 2:
    #         plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "g")

    #     else:
    #         plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "*", color = "b")
    
    # plt.title("SVM with linear kernal")
    # plt.xlabel("x data")
    # plt.ylabel("y data")
    # plt.show()

    #computing using radial basis function kernal
    rbfModel = svm.SVC(kernel = "rbf")
    rbfModel.fit(xTrain, yTrain)
    rbfPredict = rbfModel.predict(xTrain[:, 0:2])

    # for i in range(rbfPredict.shape[0]):
    #     if rbfPredict[i] == 1:
    #         plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "x", color = "r")
        
    #     elif rbfPredict[i] == 2:
    #         plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "g")

    #     else:
    #         plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "*", color = "b")
    
    # plt.title("SVM with Radial Kernal")
    # plt.xlabel("x data")
    # plt.ylabel("y data")
    # plt.show()

    # print(f"Linear Kernel = {(linearModel.score(xTrain, yTrain))}")
    # print(f"RBF Kernel= {(rbfModel.score(xTrain, yTrain))}")


    print(f"Linear Kernel = {(linearModel.score(xTest, yTest))}")
    print(f"RBF Kernel = {(rbfModel.score(xTest, yTest))}")

    cm = confusion_matrix(yTrain, linPredict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(yTrain))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Linear Kernel CM')
    plt.show()

    cm = confusion_matrix(yTrain, rbfPredict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(yTrain))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('RBF Kernel CM')
    plt.show()
    return

# function for implementing a decision tree classifier
def plotDecisionTree(xTrain, yTrain, xTest, yTest):
    treeModel = DecisionTreeClassifier(criterion="gini")
    treeModel.fit(xTrain, yTrain)
    pred = treeModel.predict(xTrain)
    print(f"Gini Impurity = {(treeModel.score(xTrain, yTrain))}")

    cm = confusion_matrix(yTrain, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(yTrain))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Decision Tree CM')
    plt.show()
    return

# main code execution
if __name__ == "__main__":
    data = np.array(processedData)
    for i in range(data.shape[0]):
        if categoryFreq[data[i, 2]] < 30:
            data[i, 2] = "Miscellaneous"

    twoClassData = []
    for i in range(data.shape[0]):
        if data[i, 2] == "Miscellaneous" or data[i, 2] == "RPI RATHSKELLAR 2        TROY         NY":
            twoClassData.append(data[i, :])

    maxLabel = -1
    for i in range(data.shape[0]):
        if data[i, 2] != "Miscellaneous":
            label = data[i, 2]
            data[i, 2] = classLabels[label]
            if maxLabel < classLabels[label]: maxLabel = classLabels[label]
    
    for i in range(data.shape[0]):
        if data[i, 2] == "Miscellaneous":
            data[i, 2] = maxLabel + 1

    # shuffled = np.array(twoClassData)
    # # rng = np.random.default_rng()
    # # permuted = rng.permutation(shuffled,axis=0)
    # permuted = np.copy(shuffled)
    # np.random.shuffle(permuted)
    # plotpermuted = permuted[:,0:2].astype(np.float64)

    # split = int(shuffled.shape[0]*.8)
    # inputTrain = shuffled[:split, 0:2]
    # inputTest = shuffled[split:, 0:2]
    # labelTrain = shuffled[:split,2]
    # labelTest = shuffled[split:2]
    inputTrain, inputTest, labelTrain, labelTest = train_test_split(data[:, 0:2], data[:, 2], test_size=0.2, random_state=None)
    # inputScalar = preprocessing.StandardScaler().fit_transform(inputTrain)
    plotKMeans(inputTrain, labelTrain, inputTest, labelTest, 5)
    plotSVM(inputTrain, labelTrain, inputTest, labelTest)
    plotDecisionTree(inputTrain, labelTrain, inputTest, labelTest)
    