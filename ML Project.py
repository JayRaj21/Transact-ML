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


"""

#interpret the data
RAW_DATA = pd.read_csv("credit_transactions.csv")
TRANSACT_DATA = RAW_DATA[RAW_DATA["Amount"]<=0]
TRANSACT_DATA["Amount"] = TRANSACT_DATA["Amount"] * -1
allData = [[], [], [], [], [], [], []]
weekNum, amountList = list(), list()

for i in TRANSACT_DATA["Effective Date"]:
    weekNum.append(dt.datetime.strptime(i, '%m/%d/%Y').isoweekday())

for i in TRANSACT_DATA["Amount"]:
    amountList.append(i)

for i in range(len(weekNum)):
    allData[weekNum[i]-1].append(amountList[i])

for i in allData:
    print(i)


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

if __name__ == "__main__":
    pass
