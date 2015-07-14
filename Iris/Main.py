__author__ = 'Giovanni'
import LoadData
import LogisticRegr
import theano
import theano.tensor as T
import numpy as np

#batch_size = 10
learning_rate=0.013
#index = T.lscalar()  # index to a [mini]batch

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels


print '... loading data'
dataSets = LoadData.LoadData('DataSet/IrisCorpus.data')

train_set_x, train_set_y = dataSets[0]
valid_set_x, valid_set_y = dataSets[1]
test_set_x, test_set_y = dataSets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]

print '... building the model'



classifier = LogisticRegr.LogisticRegr(input=x, n_in=4, n_out=3)
cost = classifier.negative_log_likelihood(y)

#y = train_set_y[: n_train_batches]
#x = train_set_x[: n_train_batches]


#numErrorSim = classifier.errors(y)
#numErrorFunc = theano.function([], numErrorSim)


test_model = theano.function(
    inputs=[],
    outputs=classifier.errors(y),
    givens={
        x: test_set_x[: n_test_batches],
        y: test_set_y[: n_test_batches]
    }
)

validate_model = theano.function(
    inputs=[],
    outputs=classifier.errors(y),
    givens={
        x: valid_set_x[: n_valid_batches],
        y: valid_set_y[: n_valid_batches]
    }
)

# compute the gradient of cost with respect to theta = (W,b)
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

updates = [(classifier.W, classifier.W - learning_rate * g_W),
            (classifier.b, classifier.b - learning_rate * g_b)
          ]

train_model = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[: n_train_batches],
            y: train_set_y[: n_train_batches]
        }
    )

predict_model = theano.function(
    inputs=[x],
    outputs=classifier.y_pred)

numErrorsInTest = test_model()
costValue = train_model()

for index in xrange(1900):
    costValue = train_model()

numErrorsInTest = test_model()

for index in xrange(2000):
    costValue = train_model()

numErrorsInTest = test_model()
numErrorsInVal = validate_model()

#for index in xrange(6000):
#    costValue = train_model()

#numErrorsInTest = test_model()
#numErrorsInVal = validate_model()



values_validDataSet = valid_set_x.get_value()
yToPrintS =  valid_set_y
yToPrintF = theano.function([],yToPrintS)
yToPrintV = yToPrintF()

predicted_values = predict_model(values_validDataSet)
print values_validDataSet
print "     True Values: " + str(yToPrintV)
print "Predicted Values: " + str(predicted_values)
print 'Finish'

#numErrorsRes = numErrorFunc()
#print ('... training the model, num Errors: %i' , numErrorsRes)


#numErrorsRes = numErrorFunc()
#print ('... training the model, num Errors: %i' , numErrorsRes)

#numErrorsRes = numErrorFunc()
#print ('... training the model, num Errors: %i' , numErrorsRes)


print ('Ok Main')