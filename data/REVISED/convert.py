import csv
import ast
''' CONVERTS TXT FILE TO CSV '''
data1 = []
for line in open('training_tweets.txt'):
    data1.append(line)  
data2 = []
for line in open('testing_tweets.txt'):
    data2.append(line)   

dates = []
scores = []
for line in data2:
	tweet = ast.literal_eval(line)
	date = tweet[0]
	dates.append(date)
	score = tweet[1]
	scores.append(score)
for line in data1:
	tweet = ast.literal_eval(line)
	date = tweet[0]
	dates.append(date)
	score = tweet[1]
	scores.append(score)	
dates.reverse()
scores.reverse()

list1 = dates
list2 = scores
rows = zip(list1,list2)
with open('tweets.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    for row in rows:
        a.writerow(row)