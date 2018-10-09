import pandas


readData = pandas.read_csv("adult.csv", skipinitialspace=True, names=[
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'grade-number',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'wage'
])

print readData