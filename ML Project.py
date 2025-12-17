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

def get_week_number(dt):
# Getting the year and week number using iso calendar
    return dt.isocalendar()[0], dt.isocalendar()[1]

# Create a dictionary to hold the frequency count per week for each category
weekly_frequency = defaultdict(lambda: defaultdict(int))

# Process each spending category
for category, timestamps in categorizedDates.items():
    for timestamp in timestamps:
        year, week = get_week_number(timestamp)

        weekly_frequency[category][(year, week)] += 1

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
    
for category, weeks in weekly_frequency.items():
   for key, freq in weeks.items():
        weeklyMeansVar[category][key].append(freq)

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

# aggregates data in format that can be accepted by scikit-learn algos
processedData = []
for i, j in weeklyMeansVar.items():
    temp = []
    for k, v in j.items():
        processedData.append([float(v[0]), float(v[1]), int(v[2]), i])

# function for implementing a k means clustering
def plotKMeans(xTrain, yTrain, xTest, yTest, k):
    model = KMeans(n_clusters=k)
    model.fit(xTrain, yTrain)
    predictor = model.fit_predict(xTrain, yTrain)

    for i in range(predictor.shape[0]):
        if predictor[i] == 1:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "r")
        
        elif predictor[i] == 2:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "g")

        elif predictor[i] == 3:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "y")

        elif predictor[i] == 4:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "m")

        elif predictor[i] == 5:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "k")
    
    plt.title("K Means")
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.show()

    print(f"Train SSE = {-(model.score(xTrain, yTrain))}")
    print(f"Test SSE = {-(model.score(xTest, yTest))}")
    return

# function for implementing SVM classifier
def plotSVM(xTrain, yTrain, xTest, yTest, cm = False):
    #computing using linear kernal
    linearModel = svm.SVC(kernel = "linear")
    linearModel.fit(xTrain, yTrain)
    linPredict = linearModel.predict(xTrain)

    for i in range(linPredict.shape[0]):
        if linPredict[i] == "1":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "r")
        
        elif linPredict[i] == "2":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "g")

        elif linPredict[i] == "3":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "y")

        elif linPredict[i] == "4":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "m")

        else:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "k")
    
    plt.title("SVM with linear kernal")
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.show()

    #computing using radial basis function kernal
    rbfModel = svm.SVC(kernel = "rbf")
    rbfModel.fit(xTrain, yTrain)
    rbfPredict = rbfModel.predict(xTrain[:, 0:2])

    for i in range(rbfPredict.shape[0]):
        if rbfPredict[i] == "1":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "r")
        
        elif rbfPredict[i] == "2":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "g")

        elif rbfPredict[i] == "3":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "y")

        elif rbfPredict[i] == "4":
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "m")

        else:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "k")
    
    plt.title("SVM with RBF kernal")
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.show()

    print(f"Linear Kernel (Train) = {(linearModel.score(xTrain, yTrain))}")
    print(f"Linear Kernel (Test) = {(linearModel.score(xTest, yTest))}")
    print(f"RBF Kernel (Train) = {(rbfModel.score(xTrain, yTrain))}")
    print(f"RBF Kernel (Test) = {(rbfModel.score(xTest, yTest))}")

    # display confusion matrix for class assignments
    if cm:
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
def plotDecisionTree(xTrain, yTrain, xTest, yTest, cm = False):
    treeModel = DecisionTreeClassifier(criterion="gini")
    treeModel.fit(xTrain, yTrain)
    pred = treeModel.predict(xTrain)
    print(f"Gini Impurity (Train) = {(treeModel.score(xTrain, yTrain))}")
    print(f"Gini Impurity (Test) = {(treeModel.score(xTest, yTest))}")

    for i in range(pred.shape[0]):
        if int(pred[i]) == 1:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "r")
        
        elif int(pred[i]) == 2:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "g")

        elif int(pred[i]) == 3:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "y")

        elif int(pred[i]) == 4:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "m")

        elif int(pred[i]) == 5:
            plt.scatter(x=xTrain[i, 0], y = xTrain[i, 1], marker = "o", color = "k")
    
    plt.title("Decision Tree")
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.show()

    # display confusion matrix for class assignments
    if cm:
        cm = confusion_matrix(yTrain, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(yTrain))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Decision Tree CM')
        plt.show()
    return

# main code execution
if __name__ == "__main__":
    sample1 = np.array([[sub_arr[0], sub_arr[1], sub_arr[3]] for sub_arr in processedData])
    sample2 = np.array([[sub_arr[0], sub_arr[2], sub_arr[3]] for sub_arr in processedData])
    for i in range(sample1.shape[0]):
        if categoryFreq[sample1[i, 2]] < 30:
            sample1[i, 2] = "Miscellaneous"
            sample2[i, 2] = "Miscellaneous"

    maxLabel = -1
    for i in range(sample1.shape[0]):
        if sample1[i, 2] != "Miscellaneous":
            label = sample1[i, 2]
            sample1[i, 2] = classLabels[label]
            sample2[i, 2] = classLabels[label]
            if maxLabel < classLabels[label]: maxLabel = classLabels[label]
    
    for i in range(sample1.shape[0]):
        if sample1[i, 2] == "Miscellaneous":
            sample1[i, 2] = maxLabel + 1
            sample2[i, 2] = maxLabel + 1

    SHOW_CM = False
    data = sample1
    inputTrain, inputTest, labelTrain, labelTest = train_test_split(data[:, 0:2], data[:, 2], test_size=0.25, random_state=None)
    
    plotKMeans(inputTrain, labelTrain, inputTest, labelTest, 5)
    plotSVM(inputTrain, labelTrain, inputTest, labelTest, SHOW_CM)
    plotDecisionTree(inputTrain, labelTrain, inputTest, labelTest, SHOW_CM)
