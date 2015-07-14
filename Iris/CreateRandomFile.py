__author__ = 'Giovanni'
import random

def MakeRowsRandom(sourceName,targetName):
    with open(sourceName,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(targetName,'w') as target:
        for _, line in data:
            target.write( line )

MakeRowsRandom('DataSet\iris_Random.data','DataSet\iris_Random2.data')