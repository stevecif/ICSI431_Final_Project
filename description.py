''' Time for some psuedo code fam.

We want to know what the probability would be to see an increase in the value of a stock the same day twitter sentiment related to that stock increases in positive sentiment, or, conversely, the probability we would see a decrease in the value of the stock the same day the twitter sentiment related to that stock dropped in sentiment.

Using Tweet(date, sentiment score) and Stock(date, closing price) from the year 2013 as our training set, and Tweet(date, sentiment score) and Stock(date, closing price) from the year 2014 as our testing set, the simple project is to measure the accuracy of such a correlation, which would serve to give all those interested a better understanding of how useful Twitter sentiment data might be for making predictions about the stock market. 
'''

print " "

# Example problem
stock_data = [['1/1/2013', 80.00],['1/2/2013', 81.00],['1/3/2013', 84.00],['1/4/2013', 79.00],['1/5/2013', 80.00]]
tweet_data = [['1/1/2013', 0.62],['1/2/2013', 0.72],['1/3/2013', 0.78],['1/4/2013', 0.82],['1/5/2013', 0.62]]

dates = []
prices = []
scores = []
for x in range(len(stock_data)):
    date = stock_data[x][0]
    dates.append(date)
    price = stock_data[x][1]
    prices.append(price)
    score = tweet_data[x][1]
    scores.append(score)

# This next block creates a list of the dates where the price either rose or fell from the previous day.
sto_ups = []
sto_downs = []
for x in range(len(prices)):
    if x + 1 < len(prices):
        if prices[x+1] > prices[x]:
            sto_ups.append(dates[x+1])
        elif prices[x+1] < prices[x]:
            sto_downs.append(dates[x+1])

print "The days where the price increased are ", sto_ups
print "The days where the price decreased are ", sto_downs

# This next block creates a list of the dates where the twitter sentiment either rose or fell from the previous day.
twe_ups = []
twe_downs = []
for x in range(len(scores)):
    if x + 1 < len(scores):
        if scores[x+1] > scores[x]:
            twe_ups.append(dates[x+1])
        elif scores[x+1] < scores[x]:
            twe_downs.append(dates[x+1])

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






