__author__ = 'Giovanni'

import numpy

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
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
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

class TestLogisticRegression(object):
    def __init__(self, input, n_in, n_out):


        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

    def negative_log_likelihood(self, r):
        return r * 1000



index = T.lscalar()  # index to a [mini]batch

x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

matrix_X = numpy.matrix([[1, 1 ,1 ,1 ],[2, 2 ,2 ,2 ],[3, 3 ,3 ,3 ],[4, 4 ,4 ,4 ],[5, 5 ,5 ,5 ]])
vector_Y = numpy.array([  2., 2. ,2. ,2. ,2.  ]) #numpy.ndarray(shape=(5,), buffer=numpy.array([2.0 ,2.0 ,2.0 ,2.0 ,2.0 ]), dtype=long)

shared_x = theano.shared(numpy.asarray(matrix_X,
                                               dtype=theano.config.floatX),
                                 borrow=True)
shared_y = theano.shared(numpy.asarray(vector_Y,
                                               dtype=theano.config.floatX),
                                 borrow=True)
shared_y_Int = T.cast((shared_y * index), 'int32')
shared_x_Int = T.cast((shared_x * index), 'int32')


classifier = TestLogisticRegression(input=x, n_in=5, n_out=4)


cost = classifier.negative_log_likelihood(y)

updates = [(classifier.W, classifier.W + x),
            (classifier.b, classifier.b * 5)]

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        y: shared_y_Int,
        x: shared_x
    },
    #on_unused_input='ignore'
)

resul1 = train_model(1)
result2 = train_model(2)
result3 = train_model(3)
result4 = train_model(2)

print("Finish")