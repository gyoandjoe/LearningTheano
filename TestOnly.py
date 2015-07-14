__author__ = 'Giovanni'
import theano.tensor as T
import theano
import numpy as np


x = T.matrix('x')
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
sumRes = sum(testRes.T)

TestResRow0 = testRes[0,:]
TestResRow0Sum  = sum( testRes[0,:])

TestResRow1 = testRes[1,:]
TestResRow1Sum  = sum( testRes[1,:])

TestResRow2 = testRes[2,:]
TestResRow2Sum  = sum( testRes[2,:])
print testRes


neq1 = theano.function([], T.neq(5,10))()
neq2 = theano.function([],T.neq(3,3))()
#MaxSim = T.argmax(test, axis=1) #Regresa el indice del valor mas grande en el eje que se indique
#MaxF = theano.function([test],MaxSim)
#resMax = MaxF(testRes)

#logSim = T.log(test)
#logF = theano.function([test],logSim)
#logRes = logF(testRes)

######

TarSim = T.arange(10)
TarFun = theano.function([], TarSim)
TarRes = TarFun()


#####
sqrSim = theano.tensor.sqr(10)
sqrFun = theano.function([],sqrSim)
sqrRes = sqrFun()
