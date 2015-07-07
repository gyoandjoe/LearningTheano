__author__ = 'Giovanni'
import os
import numpy as np
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def LoadData(FilePath):
    data_dir, data_file = os.path.split(FilePath)
    if data_dir != "" and not os.path.isfile(FilePath):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            FilePath
        )
        if os.path.isfile(new_path):
            FilePath = new_path


    print '... loading data'

    # Load the dataset
    #f = open(FilePath, 'r')
    #x = f.read()
    #f.close()
    x = np.loadtxt(FilePath, delimiter=',', usecols=(0,1,2,3))
    y = np.loadtxt(FilePath, delimiter=',', dtype=np.int, usecols=([4]))

    # - 60% Train
    # - 20% vaild_set
    # - 20% test_set



    trainSet = [(x[:90,:]), (y[:90,])]
    #trainSet_y = y[:90,]

    validSet = [(x[90:120,:]), (y[90:120,])]
    #validSet_y = y[90:120,]

    testSet = [(x[120:150,:]), (y[120:150,])]
    #testSet_y = y[120:150,]

    trainSet_x, trainSet_y = shared_dataset(trainSet)
    validSet_x, validSet_y = shared_dataset(validSet)
    testSet_x, testSet_y = shared_dataset(testSet)

    return [(trainSet_x,trainSet_y), (validSet_x,validSet_y), (testSet_x,testSet_y) ]

#rr = LoadData('DataSet/IrisCorpus.data')
#print ("OKOK")