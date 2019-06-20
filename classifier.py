import csv
from pandas import DataFrame
import pandas as pd

column = ['Title','Body']
dfBA = DataFrame(columns=column)
dfT = DataFrame(columns=[0, 1, 2, 3, 4])


def create_data_frame(data):
    t = {}
    d = {
        'Title': data[1],
        'Body': data[0],
    }

    for n in range(5):
        if len(data[2]) > n:
            t[n] = [data[2][n]]
        else:
            t[n] = ['0']
        

        
        # df = DataFrame(data=d)

        # global dfBA, dfT

        # dfT = dfT.append(df)

def parser(path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        tags = []
        data = []

        for row in reader:
            tags = tuple([x for x in row[2:-1] if x != ''])
            terms = [row[0], row[1], tags]
            data.append(terms)
            create_data_frame(terms)

if __name__ == '__main__':
    path = 'data/vzn/TrainingData.csv'
    parser(path)
    print(dfT)
    df = pd.concat([dfBA, dfT], axis=1)