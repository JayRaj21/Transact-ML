from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.mixture as mix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#interpret the data
RAW_DF_CSV = pd.read_csv("credit_transactions.csv")
NEG_DF = RAW_DF_CSV[RAW_DF_CSV["Amount"]<=0]
classDict = dict()

# print(RAW_DF_CSV.info())
# print(NEG_DF["Extended Description"])

# NEG_DF['Amount'].plot(kind='hist', bins=50, edgecolor='black', alpha=0.7)
# plt.title('Transactions Distribution')
# plt.xlabel('Amount')
# plt.ylabel('Frequency')

# NEG_DF.plot(kind="scatter", color="g", x="Effective Date", y="Amount")
# plt.title('Transactions for a year')
# plt.xlabel('Effective Date')
# plt.ylabel('Amount')

# plt.hist(classDict.values(), bins=len(classDict), color="g")
# plt.title("Number of transactions per category")
# plt.ylabel("Frequency")
# plt.xlabel("Spending Categories")
# plt.show()

NEG_DF["Date Time"] = pd.to_datetime(NEG_DF["Effective Date"], format=('%m/%d/%Y'))

def processData(data):
    for i in NEG_DF["Extended Description"]:
        if (i not in classDict.keys()):
            classDict[i] = 1

        else:
            classDict[i]+=1

    for i in NEG_DF["Amount"]:
        NEG_DF["Amount"] = i * -1

    print(NEG_DF["Amount"])
    return


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

    #runs E and M step for 200 iterations or until the error reaches a set amount (1e-6)
    for i in range(200):
        #E-Step
        iter = i
        for p in range(data.shape[0]):
            members[p, :] = np.zeros(k)  #resetting cluster matrix
            minVal = np.argmin(distArr[p, :])   #finding closest value to cluster centers
            members[p, minVal] = 1  #assiging point to new cluster

        #M-Step
        for c in range(centers.shape[0]):
            avg = np.mean(data[np.argwhere(members[:, c] == 1)], axis = 0)
            centers[c] = avg

        for p in range(data.shape[0]):
            for q in range(k):
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
    # plt.show()
    return


def GMM(data, k):
    _, _, centers = kMeansClustering(data, k)
    mean, sigma = np.zeros(k), np.zeros(k)
    
    #initializes means and covariance matrices
    for i in range(centers.shape[0]):
        mean[i] = np.mean(centers[i])
        sigma[i] = np.cov(centers[i])
    
    #computes gmm using a function from the scikit-learn library
    gmm_obj = mix.GaussianMixture(n_components=k, covariance_type="full")
    gmm_obj.fit(data)
    w = gmm_obj.predict_proba(data)
    for i in range(150): pass
    plotGMM(data, w, k)
    for i in range(150): pass
    return w


def plotGMM(data, w, k):
    plt.figure()
    plt.scatter(x=data[:,0],y=data[:,1],c=w)
    plt.title(f"k = {k}")
    plt.xlabel("x data")
    plt.ylabel("y data")
    plt.show()
    return


def plotlogReg(data):
    return


def plotQuadDisc(data):
    return


if __name__ == "__main__":
    pass
processData()
