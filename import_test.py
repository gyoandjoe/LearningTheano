import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('DataSet/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()