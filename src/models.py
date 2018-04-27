from keras.layers import GlobalMaxPooling2D,multiply,Conv2D
from keras.layers import Input,BatchNormalization,Activation,Reshape,Lambda
from keras.layers import Concatenate,AveragePooling2D,Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers import Dense
from keras.applications import InceptionResNetV2
from keras.layers import MaxPooling2D
from keras.utils import get_file
from keras.engine.topology import get_source_inputs

from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import tensorflow as tf
import os
from .. import config
####################################################
########## M NET#############################
###################################################

BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'
INITIAL_WEIGHT_PATH=config.output_path()
WEIGHT_NAME="inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"

#### wanted to use but here it is not needed#############
##
### file url
##def preprocess_input(x):
##    """ preprocess a numpy array encoding a batch of images(4D)
##    #Arguments
##
##       x:a 4D numpy array consists of RGB values within [0,255]
##
##     #Returns
##       preprocessed array
##
##
##    """
##    return imagenet_utils.preprocess_input(x,mode="tf")
##def conv2d_bn(x,
##           filters,
##           kernel_size,
##           strides=1,
##           padding="same",
##           activation="relu",
##           use_bias=False,
##           name=None):
##    """utility function to apply conv+BN
##    
##    #Arguments:
##      use_bias:whether to use bias in Conv2D
##      name:name of the ops:will be 'name'+'_ac' for the activation
##      and 'name'+"_bn" for batchnormalization
##
##
##
##    #Returns
##       output tensor after applying 'Conv2D' and Batchnormalization
##
##    """
##    x=Conv2D(filters,
##             kernel_size,
##             strides=strides,
##             padding=padding,
##             use_bias=use_bias,
##             name=name)(x)
##
##    if not use_bias:
##        bn_axis=1 if K.image_data_format()=="channel_first" else 3
##        bn_name=None if name is None else name+"_bn"
##        x=BatchNormalization(axis=bn_axis,scale=False,name=bn_name)(x)
##    if activation is not None:
##        ac_name=None if name is None else name+"_ac"
##        x=Activation(activation,name=ac_name)(x)
##
##    return x
##
##
##
##
##def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
##    """Adds a Inception-ResNet block.
##
##    This function builds 3 types of Inception-ResNet blocks mentioned
##    in the paper, controlled by the `block_type` argument (which is the
##    block name used in the official TF-slim implementation):
##        - Inception-ResNet-A: `block_type='block35'`
##        - Inception-ResNet-B: `block_type='block17'`
##        - Inception-ResNet-C: `block_type='block8'`
##
##    # Arguments
##        x: input tensor.
##        scale: scaling factor to scale the residuals (i.e., the output of
##            passing `x` through an inception module) before adding them
##            to the shortcut branch. Let `r` be the output from the residual branch,
##            the output of this block will be `x + scale * r`.
##        block_type: `'block35'`, `'block17'` or `'block8'`, determines
##            the network structure in the residual branch.
##        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
##            are repeated many times in this network. We use `block_idx` to identify
##            each of the repetitions. For example, the first Inception-ResNet-A block
##            will have `block_type='block35', block_idx=0`, ane the layer names will have
##            a common prefix `'block35_0'`.
##        activation: activation function to use at the end of the block
##            (see [activations](../activations.md)).
##            When `activation=None`, no activation is applied
##            (i.e., "linear" activation: `a(x) = x`).
##
##    # Returns
##        Output tensor for the block.
##
##    # Raises
##        ValueError: if `block_type` is not one of `'block35'`,
##            `'block17'` or `'block8'`
##    """
##    if block_type=="block35":
##        branch_0=conv2d_bn(x,32,1)
##        branch_1=conv2d_bn(x,32,1)
##        branch_1=conv2d_bn(branch_1,32,3)
##        branch_2=conv2d_bn(x,32,1)
##        branch_2=conv2d_bn(branch_2,48,3)
##        branch_2=conv2d_bn(branch_2,64,3)
##        branches=[branch_0,branch_1,branch_2]
##    elif block_type=="block17":
##        branch_0=conv2d_bn(x,192,1)
##        branch_1=conv2d_bn(x,128,1)
##        branch_1=conv2d_bn(branch_1,160,[1,7])
##        branch_1=conv2d_bn(branch_1,192,[7,1])
##        branches=[branch_0,branch_1]
##    elif block_type=="block8":
##        branch_0=conv2d_bn(x,192,1)
##        branch_1=conv2d_bn(x,192,1)
##        branch_1=conv2d_bn(branch_1,224,[1,3])
##        branch_1=conv2d_bn(branch_1,256,[3,1])
##        branches=[branch_0,branch_1]
##    else:
##        raise ValueError("Unknown Inception-Resnet block type."
##                         "Expects 'block35','block17','block8'"
##                         "but got :"+str(block_type))
##
##
##    block_name=block_type+"_"+str(block_idx)
##    channel_axis=1 if K.image_data_format()=="channel_first" else 3
##    mixed=Concatenate(axis=channel_axis,name=block_name+"mixed")(branches)
##    up=conv2d_bn(mixed,
##                 K.int_shape(x)[channel_axis],
##                 1,
##                 activation=None,
##                 use_bias=True,
##                 name=block_name+"_conv"
##                 )
##    x=Lambda(lambda inputs,scale:inputs[0]+inputs[1]*scale,
##             output_shape=K.int_shape(x)[1:],
##             arguments={'scale':scale},
##             name=block_name)([x,up])
##             
##    if activation is not None:
##
##        x=Activation(activation,name=block_name+"_ac")(x)
##    return x
##
##def inceptionResnetv(weight_path=None,input_shape=None,pooling=None,classes=4):
##    
##    if input_shape is not None:
##        img_input=Input(shape=input_shape)
##    else:
##
##        img_input=Input(shape=(299,299,3))
##    #stem block:35X35X192
##    x=conv2d_bn(img_input,32,3,strides=2,padding="valid")
##    x=conv2d_bn(x,32,3,padding="valid")
##    x=conv2d_bn(x,64,3)
##    x=MaxPooling2D(3,strides=2)(x)
##    x=conv2d_bn(x,80,1,padding="valid")
##    x=conv2d_bn(x,192,3,padding="valid")
##    x=MaxPooling2D(3,strides=2)(x)
##
##    #Mixed 5b (InceptionA block:35X35X320)
##    
##
##    branch_0 = conv2d_bn(x, 96, 1)
##    branch_1 = conv2d_bn(x, 48, 1)
##    branch_1 = conv2d_bn(branch_1, 64, 5)
##    branch_2 = conv2d_bn(x, 64, 1)
##    branch_2 = conv2d_bn(branch_2, 96, 3)
##    branch_2 = conv2d_bn(branch_2, 96, 3)
##    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
##    branch_pool = conv2d_bn(branch_pool, 64, 1)
##    branches = [branch_0, branch_1, branch_2, branch_pool]
##    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
##    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)
##
##
##    #10X block35 (Inception-ResNet-A block): 35 x 35 x 320
##    
##    for block_idx in range(1,11):
##        x=inception_resnet_block(x,
##                                 scale=0.17,
##                                 block_type='block35',
##                                 block_idx=block_idx)
##    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
##    branch_0=conv2d_bn(x,384,3,strides=2,padding="valid")
##    branch_1=conv2d_bn(x,256,1)
##    branch_1=conv2d_bn(branch_1,256,3)
##    branch_1=conv2d_bn(branch_1,384,3,strides=2,padding="valid")
##    branch_pool=MaxPooling2D(3,strides=2,padding='valid')(x)
##    branches=[branch_0,branch_1,branch_pool]
##    x=Concatenate(axis=channel_axis,name="mixed_6a")(branches)
##    
##    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
##
##    for block_idx in range(1,21):
##        x=inception_resnet_block(x,
##                                 scale=0.1,
##                                 block_type="block17",
##                                 block_idx=block_idx)
##
##
##    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
##    
##
##    branch_0 = conv2d_bn(x, 256, 1)
##    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
##    branch_1 = conv2d_bn(x, 256, 1)
##    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
##    branch_2 = conv2d_bn(x, 256, 1)
##    branch_2 = conv2d_bn(branch_2, 288, 3)
##    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
##    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
##    branches = [branch_0, branch_1, branch_2, branch_pool]
##    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)
##
##    
##    for block_idx in range(1, 10):
##        x = inception_resnet_block(x,
##                                   scale=0.2,
##                                   block_type='block8',
##                                   block_idx=block_idx)
##
##
##    x = inception_resnet_block(x,
##                           scale=1.,
##                           activation=None,
##                           block_type='block8',
##                           block_idx=10)
##
##    #Final convolution block: 8X8X1536
##
##    x=conv2d_bn(x,1536,1,name="conv_7b")
##
##   
##    if pooling=="avg":
##        x=GlobalAveragePooling2D(name="avg_pool")(x)
##    else:
##        
##        x=GlobalMaxPooling2D(name="max_pool")(x)
##
##    
##    #final layer
##    
##    
##
##    model=Model(inputs=img_input,outputs=x,name="inception_resnet_v2")
##
##    #load weight
##    if weight_path is not None:
##        fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
##        model.load_weights(weight_path)
##    
##    return model




def custom_inceptionResnetV2(input_shape=config.image_size,classes=config.num_classes,is_base_trainable=False):
    base_model=InceptionResNetV2(include_top=False,weights=os.path.join(INITIAL_WEIGHT_PATH,WEIGHT_NAME),input_shape = input_shape)
    print("base_model loaded")
    base_model.trainable=is_base_trainable
    img_input = Input(shape = input_shape)
    x = base_model(img_input)
    x = AveragePooling2D((8, 8), name='avg_pool')(x)
    x = Flatten()(x)

    #building a FC model (for 2 classes) on top of the resenet-50 model
    x = Dense(5, activation='softmax', name='fc1')(x)

    model = Model(img_input, x)
    return model

##model=custom_inceptionResnetV2()
##model.summary()











###########################################################################
#####################  A NET ###########################################
######################################################################

########################################################
#lambda function for spatial softmax

def spatial_softmax_lamba(xs):
    
    d=tf.shape(xs)
    N,H,W,C=d[0],d[1],d[2],d[3]
    #Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
    x=tf.reshape(tf.transpose(xs,[0,3,1,2]),[N*C,H*W])
    softmax=tf.nn.softmax(xs)
    # Reshape and transpose back to original format.
    softmax=tf.transpose(tf.reshape(softmax,[N,C,H,W]),[0,2,3,1])
    return softmax

#attention map
def attentionmap():
    #part-1
    x=Input(shape=(14,14,1024))
    X=Conv2D(filters=5,kernel_size=1,name="a-net-part-1")(x)
    
    #part-2
    #block-1
    Y=Conv2D(1024,3,padding="same",name="block-1-a-net-part-2")(x)
    Y=BatchNormalization(axis=3,scale=False,name="bn_part_1")(Y)
    Y=Activation("relu")(Y)
    #block-2
    Y=Conv2D(1024,3,padding="same",name="block-2-a-net-part-2")(Y)
    Y=BatchNormalization(axis=3,scale=False,name="bn_part_2")(Y)
    Y=Activation("relu")(Y)

    #block-3
    Y=Conv2D(5,1,padding="same",name="block-3-a-net-part-2")(Y)
    Y=Activation("relu")(Y)
    #spatial-softmax
    Y=Lambda(spatial_softmax_lamba,output_shape=(14,14,5))(Y)

    #element wise multiplication merge
    Y=multiply([X,Y])
    Y=GlobalMaxPooling2D()(Y)
    model=Model(inputs=x,outputs=Y)
    return model





#model=attentionmap()
#model.compile(optimizer="rmsprop",loss="categorical_crossentropy",
              #metrics=['accuracy'])
#model.summary()

#M-net
##model=inceptionResnetv()



