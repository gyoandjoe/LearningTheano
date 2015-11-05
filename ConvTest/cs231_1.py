__author__ = 'Giovanni'
import numpy as np
import pylab
from PIL import Image
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor4(name='input')

no_filters = 2
size_filter = 3

depth_in = 3
size_in = 5
# initialize shared variable for weights.
w_shp = (no_filters, depth_in, size_filter, size_filter)
b_shp = (2, 1, 1, 1)
filter_bias = numpy.array([[[1]],[[0]]] )

filter_conv = numpy.array([[
    [[0,1,-1],
    [1,1,-1],
    [0,0,1]],
    [[0,0,0],
    [0,-1,1],
    [-1,0,1]],
    [[1,-1,1],
    [-1,1,0],
    [-1,-1,0]]
    ],[
    [[0,1,0],
    [1,1,0],
    [1,-1,1]],
    [[-1,0,-1],
    [1,1,0],
    [-1,-1,-1]],
    [[0,-1,0],
    [0,-1,0],
    [0,0,-1]]
    ]],dtype=input.dtype)

W = theano.shared(filter_conv, name ='W')


# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv.conv2d(input=input, filters= W, subsample=(2,2), border_mode ='valid') + filter_bias

#output = T.nnet.sigmoid(conv_out)

# create theano function to compute filtered images
f = theano.function([input], conv_out)


matriz_in = numpy.array([[
    [
    [0,0,0,0,0,0,0],
    [0,0,0,2,1,2,0],
    [0,0,0,1,1,2,0],
    [0,0,2,1,2,1,0],
    [0,0,1,1,2,2,0],
    [0,1,2,1,2,1,0],
    [0,0,0,0,0,0,0]
    ],[
    [0,0,0,0,0,0,0],
    [0,0,0,1,0,1,0],
    [0,2,0,0,2,0,0],
    [0,1,2,0,2,0,0],
    [0,0,0,2,1,0,0],
    [0,1,1,0,1,2,0],
    [0,0,0,0,0,0,0]
    ],[
    [0,0,0,0,0,0,0],
    [0,2,2,0,1,1,0],
    [0,2,1,2,2,1,0],
    [0,1,0,1,1,0,0],
    [0,2,0,0,1,0,0],
    [0,2,2,2,1,0,0],
    [0,0,0,0,0,0,0]
    ]
    ]],dtype=theano.config.floatX)

#mm = matriz_in[0][0]
#paddok = numpy.lib.pad(mm   , 2, 0)
resfull=f(matriz_in)
resu =  resfull[0, 0, :,:]

# I = numpy.asarray(PIL.Image.open('test.jpg'))
# open random image of dimensions 639x516
img = Image.open('E:\dev\DeepLearningTutorial\DataSet\Jesus.jpg')
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype=theano.config.floatX)
img =  img / 256

# put image in 4D tensor of shape (1, 3, height, width)
# con transpose queda de 3 x 678 x 1000
img_ = img.transpose(2, 0, 1)
img_= img_.reshape(1, 3, 678, 1000)

# allow_input_downcast=True
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()



#test_model = theano.function(
#    inputs=[],
#    outputs=classifier.errors(y),
#    givens={
#        x: test_set_x[: n_test_batches],
#        y: test_set_y[: n_test_batches]
#    }
#)
