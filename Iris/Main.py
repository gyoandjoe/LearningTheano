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




#mm = train_set_x.get_value(borrow=True)
#W = theano.shared(
#    value=np.ones(
#        (90, 4),
#        dtype=theano.config.floatX
#        ),
#        name='W',
#        borrow=True
#    )
#num1 = T.scalar()
#num2 = T.scalar()
#Wm = T.mul(x,num2)
#wres = theano.function([x,num2],Wm)
#rr = wres(train_set_x.get_value(),5)

############
testMat = np.array([[1000,815,815,815,815],[20,20,20,20,21],[0.5,0.7,0.4,0.6,0.1],[10,10,70,5,5]],dtype=theano.config.floatX)
test = T.nnet.softmax(x)
testF = theano.function([x],test)
testRes = testF(testMat)

#MaxSim = T.argmax(test, axis=1)
#MaxF = theano.function([test],MaxSim)
#resMax = MaxF(testRes)

logSim = T.log(test)
logF = theano.function([test],logSim)
logRes = logF(testRes)


print ('Ok Main')