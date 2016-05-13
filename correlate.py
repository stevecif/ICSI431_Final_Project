''' Time for some correlation code fam.

We want to know what the probability would be to see an increase in the value of a stock the same day twitter sentiment related to that stock increases in positive sentiment, or, conversely, the probability we would see a decrease in the value of the stock the same day the twitter sentiment related to that stock dropped in sentiment.

Using Tweet(date, sentiment score) and Stock(date, closing price) from the year 2013 as our training set, and Tweet(date, sentiment score) and Stock(date, closing price) from the year 2014 as our testing set, the simple project is to measure the accuracy of such a correlation, which would serve to give all those interested a better understanding of how useful Twitter sentiment data might be for making predictions about the stock market. 
'''

import csv
import random
import math
import codecs
import ast
import pandas as pd

print " "

# Pulls in the stock market data CSV (excel file) and stores the date information into "dates" array, and the closing price information into "prices" array.
df = pd.read_csv('./data/training_stock.csv')
dates = df['Date']
prices = df['Price']  
stock_data = [list(i) for i in zip(dates, prices)] 
# Pulls in Tweet data, which is strictly the date and sentiment score.
tweet_data = []
for line in open('./data/training_tweets.txt'):
    tweet_data.append(line)    
#[list(i) for i in zip(dates, scores)] 

# Refines imported data and splits it into three arrays by type, which are "dates", "prices", and "scores".
s_dates = []
prices = []
for x in range(len(stock_data)):
    date = stock_data[x][0]
    s_dates.append(date)
    price = stock_data[x][1]
    prices.append(price)

t_dates = []
scores = []
# for x in range(len(tweet_data)):
#     date = tweet_data[x][0]
#     t_dates.append(date)
#     score = tweet_data[x][1]
#     scores.append(score)

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
for x in range(len(prices)):
    if x + 1 < len(prices):
        if prices[x+1] > prices[x]:
            sto_ups.append(s_dates[x+1])
        elif prices[x+1] < prices[x]:
            sto_downs.append(s_dates[x+1])

print "The days where the price increased are ", sto_ups
print "The days where the price decreased are ", sto_downs

# This next block creates a list of the dates where the twitter sentiment either rose or fell from the previous day.
twe_ups = []
twe_downs = []
for x in range(len(scores)):
    if x + 1 < len(scores):
        if scores[x+1] > scores[x]:
            twe_ups.append(t_dates[x+1])
        elif scores[x+1] < scores[x]:
            twe_downs.append(t_dates[x+1])

print "The days where the sentiment increased are ", twe_ups
print "The days where the sentiment decreased are ", twe_downs
print " "

a = sto_ups
b = twe_ups
c = sto_downs
d = twe_downs
up_matches = [i for i, j in zip(a, b) if i == j]
down_matches = [i for i, j in zip(c, d) if i == j]
correlation1 =  float(len(up_matches)) / float(len(dates))
correlation2 =  float(len(down_matches)) / float(len(dates))
print "The correlation score for tweets and stock increases in this data set is ", correlation1
print "The correlation score for tweets and stock decreases in this data set is ", correlation2






