__author__ = 'Giovanni'
import theano
import theano.tensor as T
import numpy as np

class LogisticRegr(object):

    #n_in  = No de Variables
    #n_out = No de clases
    #input = DataSet sin valores de respuesta
    def __init__(self, input, n_in, n_out):


        self.input = input
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        #Da la probabilidad de cada clase
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #Da la mejor probabilidad entre todas las clases
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return T.sum( 1 - self.p_y_given_x[T.arange(y.shape[0]), y] )


    def errors(self, y):
        """ T.neq = el primer parametro es diferente del segundo """
        return T.sum(T.neq(self.y_pred, y))


