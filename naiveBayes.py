''' Time for some naive bayes action fam.

We want to know what the probability would be to see an increase in the value of a stock the same day twitter sentiment related to that stock increases in positive sentiment, or, conversely, the probability we would see a decrease in the value of the stock the same day the twitter sentiment related to that stock dropped in sentiment.

Using Tweet(date, sentiment score) and Stock(date, closing price) from the years 2013 and 2014 as our testing and training sets, the simple project is to measure the accuracy of such a correlation, which would serve to give all those interested a better understanding of how useful Twitter sentiment data might be for making predictions about the stock market. 
'''

# This is the module where all the necessary libraries are imported.
import csv
import random
import math
import codecs
import ast
import time
from datetime import datetime
import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import pandas as pd
print " "

# These are the functions needed to make the whole Naive Bayes algorithm fire.
def csvLoad(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitData(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separate(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated    

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stanDev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-0)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stanDev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries  

def summarizeClass(dataset):
    separated = separate(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries      

def calcProb(x, mean, stanDev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stanDev,2) + 0.01)))
    return (1 / (math.sqrt(2*math.pi) * stanDev + 0.01)) * exponent

def calcClassProbs(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stanDev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calcProb(x, mean, stanDev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calcClassProbs(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# Pulls in the stock market data CSV (excel file) and stores the date information into "dates" array, and the closing price information into "prices" array.
df = pd.read_csv('./data/stocks.csv')
dates = df['Date']
prices = df['Price']  
stock_data = [list(i) for i in zip(dates, prices)] 
stock_data = stock_data
# Pulls in Tweet data, which is strictly the date and sentiment score.
tweet_data = []
for line in open('./data/tweets.txt'):
    tweet_data.append(line)    
tweet_data = tweet_data

# Refines imported data and splits it into four arrays by type, which are "t_dates", "s_dates", "prices", and "scores".
s_dates = []
prices = []
for x in range(len(stock_data)):
    date = stock_data[x][0]
    s_dates.append(date)
    price = stock_data[x][1]
    prices.append(price)
t_dates = []
scores = []
for line in tweet_data:
	tweet = ast.literal_eval(line)
	date = tweet[0]
	t_dates.append(date)
	score = tweet[1]
	scores.append(score)
t_dates.reverse()
scores.reverse()

print " "

# This next block creates a list of the dates where the price either rose or fell from the previous day.
sto_ups = []
sto_downs = []
sto_changes = []
for x in range(len(prices)):
    if x + 1 < len(prices):
        if prices[x+1] > prices[x]:
            change = float(prices[x+1]) - float(prices[x])
            sto_changes.append(change)
            sto_ups.append(s_dates[x+1])
        elif prices[x+1] < prices[x]:
            change = float(prices[x]) - float(prices[x+1])
            sto_changes.append(change)
            sto_downs.append(s_dates[x+1])

avg_sto_change = mean(sto_changes)

# This next block creates a list of the dates where the twitter sentiment either rose or fell from the previous day.
twe_ups = []
twe_downs = []
twe_changes = []
for x in range(len(scores)):
    if x + 1 < len(scores):
        if scores[x+1] > scores[x]:
            change = float(scores[x+1]) - float(scores[x])
            twe_changes.append(change)
            twe_ups.append(t_dates[x+1])
        elif scores[x+1] < scores[x]:
            change = float(scores[x]) - float(scores[x+1])
            twe_changes.append(change)
            twe_downs.append(t_dates[x+1])

avg_twe_change = mean(twe_changes)

# This block records the number of dates where prices and tweet scores rise or fall, and when they coincide. 
a = sto_ups
b = twe_ups
c = sto_downs
d = twe_downs
up_matches = list(set(a) & set(b)) 
down_matches = list(set(c) & set(d)) 

# This section calculates how often the price and stock markers coincide out of the total number of days period, and these calculations are printed accordingly.
correlation1 =  float(len(up_matches)) / float(len(s_dates))
correlation2 =  float(len(down_matches)) / float(len(t_dates))

print "The average day-to-day change in stock price was: ", avg_sto_change
print "The average day-to-day change in Twitter sentiment was: ", avg_twe_change
print "The correlation score for tweets and stock increases in this data set is ", correlation1
print "The correlation score for tweets and stock decreases in this data set is ", correlation2
print " "

'''

Alright y'all, let's take a moment to stop and think about what we've created so far. So we have a couple sets of good numbers (the changes in stock price from day to day, the changes in Twitter score sentiment from day to day, the days where prices go up or down, the days where sentiment score goes up or down), and from these we have calculated that the probability we should expect to see a correlation between these two datasets when there is an increase is roughly 20%, and the probability when there is a decrease is roughly 15%. Not super amazing numbers, but there is always room for improvement, especially once we implement an actual predictive algorithm, as opposed to merely dividing out averages.

'''

# This section revises the dates (which are stored as strings in their respective arrays), and converts them into datetime objects (which are floats).
new_sd = []
for x in s_dates:
    date_obj = datetime.strptime(x, '%Y-%m-%d')
    new_sd.append(date_obj)

new_td = []
for x in t_dates:
    date_obj = datetime.strptime(x, '%Y-%m-%d')
    new_td.append(date_obj) 

sc_class = []
pos = 1
neg = 0
for x in scores:
    if x > avg_sto_change:
        sc_class.append(pos)
    else:
        sc_class.append(neg)    

# This module creates an ID for the stocks based on their corresponding date.
def to_integer(dt_time):
    return 10000*dt_time.year + 1000*dt_time.month + dt_time.day
ids = []
for x in new_sd:
    fixed = to_integer(x)
    ids.append(fixed)

# This module writes the testing stock date/ID, the day to day change from each date, and the day-to-day changes of the sentiment scores to a CSV.
# The value of "C" is used to cull the datasets down to a mangeable size.
C = 500
list1 = ids[:C]
list2 = sto_changes[:C]
list3 = twe_changes[:C]
rows = zip(list2,list3)
with open('csv_train.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    for row in rows:
        a.writerow(row)

# Now that we've constructed a CSV from the day-to-day changes
filename = 'csv_train.csv'
splitRatio = 0.67
dataset = csvLoad(filename)
trainingSet, testSet = splitData(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
# prepare model
summaries = summarizeClass(trainingSet)
# test model
predictions = getPredictions(summaries, testSet)
print('Predictions: {0}').format(predictions)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}%').format(accuracy)


