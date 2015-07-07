__author__ = 'Giovanni'
import random

def MakeRowsRandom(fileName):
    with open(fileName,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open('DataSet\IrisCorpus.data','w') as target:
        for _, line in data:
            target.write( line )

MakeRowsRandom('DataSet\iris - norm.data')