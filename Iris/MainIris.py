__author__ = 'Giovanni'
import LoadData
import LogisticRegr
import theano
import theano.tensor as T
import numpy as np

batch_size = 10
index = T.lscalar()  # index to a [mini]batch

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels


print '... loading data'
dataSets = LoadData.LoadData('DataSet/IrisCorpus.data')

train_set_x, train_set_y = dataSets[0]
valid_set_x, valid_set_y = dataSets[1]
test_set_x, test_set_y = dataSets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

print '... building the model'

classifier = LogisticRegr.LogisticRegr(input=x, n_in=4, n_out=3)



W = theano.shared(
    value=np.zeros(
        (4, 3),
        dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

test = T.nnet.softmax(T.dot(x, W))

print ('Ok Main')