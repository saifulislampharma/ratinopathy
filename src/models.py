from keras.layers import GlobalMaxPooling2D,multiply,Conv2D
from keras.models import Sequential
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
TRAINED_WEIGHT_PATH=config.weight_path()
WEIGHT_NAME="inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
TRAINED_DENSE_WEIGHT="my_resnet_inception_v2.h5"
TRAINED_CONV_WEIGHT="my_resnet_inception_v2_conv.h5"





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


def custom_inceptionResnetV2_conv(input_shape=config.image_size,classes=config.num_classes,is_base_trainable=False):
    base_model=InceptionResNetV2(include_top=False,weights=os.path.join(INITIAL_WEIGHT_PATH,WEIGHT_NAME),input_shape = input_shape)
    print("base_model loaded")

##    model = Sequential()
##    for layer in base_model.layers:
##        model.add(layer)
##    
##    for layer in model.layers[:768]:
##       layer.trainable = False
##    for layer in model.layers[769:]:
##       layer.trainable = True
##    model.add(AveragePooling2D((8, 8), name='avg_pool'))
##    model.add(Flatten())
##    model.add(Dense(5, activation='softmax', name='fc1'))
    #setting some conv layers trainable
    base_model.trainable=True
    set_trainable=False
    for layer in base_model.layers:
        if layer.name=="conv2d_203":
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False
    
    #base_model.trainable=is_base_trainable
    img_input = Input(shape = input_shape)
    x = base_model(img_input)
    x = AveragePooling2D((8, 8), name='avg_pool')(x)
    x = Flatten()(x)

    #building a FC model (for 2 classes) on top of the resenet-50 model
    x=Dense(512,activation='relu', name='fc1_1')(x)
    x=Dense(128,activation='relu', name='fc1_2')(x)
    x = Dense(5, activation='softmax', name='fc1_3')(x)

    model = Model(img_input, x)
   # model.load_weights(os.path.join(config.weight_path(),TRAINED_DENSE_WEIGHT))
    return model


def custom_inceptionResnetV2_conv_global(input_shape=config.image_size,classes=config.num_classes,is_base_trainable=False):
    base_model=InceptionResNetV2(include_top=False,weights=os.path.join(INITIAL_WEIGHT_PATH,WEIGHT_NAME),input_shape = input_shape)
    print("base_model loaded")

##    model = Sequential()
##    for layer in base_model.layers:
##        model.add(layer)
##    
##    for layer in model.layers[:768]:
##       layer.trainable = False
##    for layer in model.layers[769:]:
##       layer.trainable = True
##    model.add(AveragePooling2D((8, 8), name='avg_pool'))
##    model.add(Flatten())
##    model.add(Dense(5, activation='softmax', name='fc1'))
    #setting some conv layers trainable
    base_model.trainable=True
    set_trainable=False
    for layer in base_model.layers:
        if layer.name=="conv2d_203":
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False
    
    #base_model.trainable=is_base_trainable
    img_input = Input(shape = input_shape)
    x = base_model(img_input)
    x = GlobalAveragePooling2D(name='gl_avg_pool')(x)
    #x = Flatten()(x)

    #building a FC model (for 2 classes) on top of the resenet-50 model
    #x=Dense(512,activation='relu', name='fc1_1')(x)
    #x=Dense(128,activation='relu', name='fc1_2')(x)
    x = Dense(5, activation='softmax', name='fc1_3')(x)

    model = Model(img_input, x)
   # model.load_weights(os.path.join(config.weight_path(),TRAINED_DENSE_WEIGHT))
    return model
##base_model=custom_inceptionResnetV2_conv()
##for i, layer in enumerate(base_model.layers):
##   print(i, layer.name)

##base_model.summary()

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



