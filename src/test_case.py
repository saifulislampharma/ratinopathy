
from keras.optimizers import SGD
from . import models
from .. import config
from . import model_utils
import os

#model=models.custom_inceptionResnetV2_conv()
##
###optimizer
##optimizer=SGD(lr=1e-3,
##              momentum=0.9,
##              nesterov=True
##
##    )
##objective="categorical_crossentropy"
##
##model.compile(optimizer=optimizer,
##              loss=objective,
##              metrics=['accuracy']
##              )
##
##
#model_utils.save_model_only(model,model_utils.MODEL_CONV)
weight_path=os.path.join(config.weight_path(),model_utils.MAIN_MODEL_WEIGHT)
model=model_utils.load_model(model_utils.MODEL_CONV,weight_path)
model.summary()
                         
##model_string=model_utils.TEST_MODEL
##model=model_utils.load_model(model_string)
###model.load_weights(os.path.join(models.TRAINED_WEIGHT_PATH,models.TRAINED_DENSE_WEIGHT))
##for layer in model.layers:
##    if layer.name=="conv2d_203":
##        for ince_layer in layer:
##            print(ince_layer)
##
##
##model.summary()
