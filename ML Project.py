from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.mixture as mix
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
Potential Classification Rules:

Group by year

Group by year and month

Group by month only

Group by week

Group by day of the week

Group amount by some threshold
    Below 10
    Between 10 and 20
    Between 20 and 30
    Between 30 and 40
    ...

Things worth considering 
    Applying PCA to the data before performing 
    any clustering or classification

Decision rule:
    Amount based rule
    y < 15

    15 < y < 30

    30 < y < 60

    60 < y < 80

    y > 80

    Next steps:
    Figure out a good number of class labels to use for my decision rule
    Figure out how to identiy probability for each category on each day
    Try using mean + c * std decision rule

    Consider identifying the likilhood of a specific category being purchased on a given week
    Consider classifying based on food types, healthiness, etc.

"""

#interpret and process the data
RAW_DATA = pd.read_csv("credit_transactions.csv")
TRANSACT_DATA = RAW_DATA[RAW_DATA["Amount"]<=0].copy()
TRANSACT_DATA["Amount"] = TRANSACT_DATA["Amount"] * -1
groupedTransacts = [[], [], [], [], [], [], []]
weekNum, amountList = list(), list()

#organizes the data set into groups of days in a week
for i in TRANSACT_DATA["Effective Date"]:
    weekNum.append(dt.datetime.strptime(i, '%m/%d/%Y').isoweekday())

for i in TRANSACT_DATA["Amount"]:
    amountList.append(i)

for i in range(len(weekNum)):
    groupedTransacts[weekNum[i] - 1].append(amountList[i])

#make the data continuous by its date so that it can be given class labels
CONTIN_DATA = TRANSACT_DATA.copy()
CONTIN_DATA["Effective Date"] = pd.to_datetime(CONTIN_DATA["Effective Date"])
CONTIN_DATA = CONTIN_DATA.groupby('Effective Date')['Amount'].sum().to_frame()
full_index = pd.date_range(CONTIN_DATA.index.min(), CONTIN_DATA.index.max(), freq='D')
CONTIN_DATA = CONTIN_DATA.reindex(full_index).fillna(0)
CONTIN_DATA['X'] = (CONTIN_DATA.index - CONTIN_DATA.index.min()).days
X, y = CONTIN_DATA[['X']].values, CONTIN_DATA['Amount'].values
LABELS = list()

# assigns relevant class labels according to transaction amount
for transac in y:
    if transac < 15:
        LABELS.append(1.0)

    elif transac >= 15 and transac < 30:
        LABELS.append(2.0)

    elif transac >= 30 and transac < 60:
        LABELS.append(3.0)

    elif transac >= 60 and transac < 80:
        LABELS.append(4.0)
    
    else:
        LABELS.append(5.0)

processedData2 = []
for i in range(X.shape[0]):
    processedData2.append([X[i][0].item(), y[i].item(), LABELS[i]])

CLASSIFY_DATA = np.array(processedData2)


# for i in allData:
#     print(i)
# print(RAW_DATA.info())
# print(TRANSACT_DATA["Extended Description"])

# TRANSACT_DATA['Amount'].plot(kind='hist', bins=50, edgecolor='black', alpha=0.7)
# plt.title('Transactions Distribution')
# plt.xlabel('Amount')
# plt.ylabel('Frequency')

# TRANSACT_DATA.plot(kind="scatter", color="g", x="Effective Date", y="Amount")
# plt.title('Transactions for a year')
# plt.xlabel('Effective Date')
# plt.ylabel('Amount')

# plt.hist(classDict.values(), bins=len(classDict), color="g")
# plt.title("Number of transactions per category")
# plt.ylabel("Frequency")
# plt.xlabel("Spending Categories")
# plt.show()

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

#function for calculating and graphing the k-means-clustering solution
#params: data (np.array), k (int)
#return: errorSum (np.array), members (np.array), centers (np.array)
def kMeansClustering(data, k, plotCheck = False):
    centers = np.random.rand(k, data.shape[-1]) #randomly generated initial cluster center
    distArr = np.full([data.shape[0], k], 0)    #distance matrix
    members = np.full([data.shape[0], k], 0)    #matrix of different clusters
    errorSum = 0                                #sum-of-squares error
    prevError = np.inf                          #previous error value
    iter = 0                                    #iteration variable for plotting

    #initiliazes distance matrix with randomly generated cluster center and data point
    for i in range(data.shape[0]):
        for j in range(k):
            distArr[i, j] = np.linalg.norm(centers[j, :] - data[i, :])
            # print(distArr[i, j])

    #runs E and M step for 200 iterations or until the error reaches a set amount (1e-6)
    for i in range(200):
        #E-Step
        iter = i
        for p in range(data.shape[0]):
            members[p, :] = np.zeros(k)  #resetting cluster matrix
            minVal = np.argmin(distArr[p, :])   #finding closest value to cluster centers
            members[p, minVal] = 1  #assiging point to new cluster
            print(minVal, members[p,: ])

        #M-Step
        for c in range(centers.shape[0]):
            # print(centers[c])
            print(members[:, c])
            # print(np.argwhere(members[:, c] == 1))
            avg = np.mean(data[np.argwhere(members[:, c] == 1)], axis = 0)
            # print(avg)
            
            
            # note that this code needs to be fixed but is working for now
            validAvg = True
            for a in avg:
                if a.all():
                    pass
                else:
                   validAvg = False 
                   break
            if not validAvg:
                centers[c] = avg
            else:
                centers[c] = 0

        for p in range(data.shape[0]):
            for q in range(k):
                # print(centers[q,:])
                # print(data[p,:])
                # print(centers[q,:]-data[p,:])
                # print(np.linalg.norm(centers[q,:]-data[p,:]))
                distArr[p, q] = np.linalg.norm(centers[q, :] - data[p, :])
        
        #computes the sum-of-squares error value
        prevError = errorSum
        errorSum = 0
        for p in range(data.shape[0]):
            for q in range(k):
                if (members[p, q] == 1):
                    sqr = (distArr[p, q])**2
                    errorSum += sqr
        
        #stops iterating if the error threshold has been reached
        if (np.abs(prevError - errorSum) < 1e-6): break

        #plots the graph for the initial, and first iterations
        if plotCheck and (i == 0 or i == 1): plotClusters(data, k, errorSum, members, centers, i)
    
    #plots clustered data for the final iteration
    if plotCheck: plotClusters(data, k, errorSum, members, centers, iter)
    return errorSum, members, centers

def plotClusters(data, k, error, members, centers, iter):
    plt.figure()
    for i in range(members.shape[-1]):
        points = np.squeeze(data[np.argwhere(members[:, i] == 1)])
        plt.scatter(points[:, 0], points[:, 1])
        plt.scatter(centers[i,0], centers[i, 1], marker = "X", color = "black")
    plt.title(f"Iteration {iter}, k = {k} --> Error = {error}")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()
    return

def plotSVM(data):
    xData, yData = data[:, 0:2], data[:, 2]

    #computing using linear kernal
    linearModel = svm.SVC(kernel = "linear")
    linearModel.fit(xData, yData)
    linPredict = linearModel.predict(data[:, 0:2])

    for i in range(linPredict.shape[0]):
        if linPredict[i] == 1:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "x", color = "r")
        
        elif linPredict[i] == 2:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "o", color = "g")

        else:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "*", color = "m")
      
    plt.title("SVM with linear kernal")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()  

    #computing using radial basis function kernal
    rbfModel = svm.SVC(kernel = "rbf")
    rbfModel.fit(xData, yData)
    rbfPredict = rbfModel.predict(data[:, 0:2])

    for i in range(rbfPredict.shape[0]):
        if rbfPredict[i] == 1:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "x", color = "r")
        
        elif rbfPredict[i] == 2:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "o", color = "g")

        else:
            plt.scatter(x=data[i, 0], y = data[i, 1], marker = "*", color = "m")
    
    plt.title("SVM with Radial Kernal")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()
    return

def plotGMM(data):
    return


def plotLogReg(data):
    xData, yData = data[:, 0:2], data[:, 2]
    logModel = LogisticRegression()
    logModel.fit(xData, yData)
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
    plt.show()
    
    #calculating cross-entropy loss and number of points missclassified
    yPred = logModel.predict_proba(data[:, 0:2])
    ceLoss = 0
    
    for i in range(yPred.shape[0]):
        ceLoss += np.log(yPred[i,int(yData[i]-1)])
    
    ceLoss *= -1
    classAcc = logModel.score(data[:, 0:2],yData)

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


if __name__ == "__main__":
    pass
    data = np.array(padData(groupedTransacts))
    # kMeansClustering(data, 5, False)
    plotSVM(CLASSIFY_DATA)
    # plotLogReg(CLASSIFY_DATA)
    
