from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,ReduceLROnPlateau
from keras.models import model_from_json
import numpy
import os
from .. import config
print("saiful")

#model name
MODEL_CONV="my_resnet_inception_v2_conv.json"
MODEL_CONV_WEIGHT="my_resnet_inception_v2_conv.h5"
MODEL_CONV_SGD_DA_WEIGHT="my_resnet_inception_v2_conv_sgd_da.h5"
MODEL_CONV_RMS_WEIGHT="my_resnet_inception_v2_conv_rms.h5"
MODEL_CONV_ADAM_WEIGHT="my_resnet_inception_v2_conv_adam.h5"
MODEL_CONV_ADAM_RL_WEIGHT="my_resnet_inception_v2_conv_adam.h5"
MODEL_CONV_ADAGRADE_WEIGHT="my_resnet_inception_v2_conv_adagrad_RL.h5"
MODEL_CONV_ADAGRADE_WEIGHT_RLF="my_resnet_inception_v2_conv_adagrad_RLF.h5"
MAIN_MODEL_WEIGHT="my_resnet_inception_v2.h5"
MAIN_MODEL="my_resnet_inception_v2.json"

#after taking all the image 0 category


MODEL_CONV_ADAM_DA_ALL_WEIGHT="my_resnet_inception_v2_conv_adam_da_all.h5"


#after adding new Dense Layer
MODEL_DENSE_ADAM_DA_ALL_WEIGHT="my_resnet_inception_v2_dense_adam_da_all.h5"

#changin to GLOBAL AVERAGE POOLING

MODEL_GAP_ADAM_DA_ALL_WEIGHT="my_resnet_inception_v2_gap_adam_da_all.h5"


def save_model(model,model_name,weight_name):
    """
    save model to output directory
    """
    
    #serialize the model to json
    model_json=model.to_json()
    #write
    with open(os.path.join(config.output_path(),model_name), "w") as json_file:
        json_file.write(model_json)
    print("model saved")

    #save the weight
    #serialize weights to HDF5
    model.save_weights(os.path.join(config.output_path(),weight_name))
    print("weight saved")
    
def load_model_only(model_name):
    """
    load a model from output directory by name
    """
    json_file=open(os.path.join(config.output_path(),model_name))
    loaded_json_file=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_json_file)
    
    return loaded_model

def load_model(model_name,weight_path):
    """
    load a model from output directory by name
    """
    json_file=open(os.path.join(config.output_path(),model_name))
    loaded_json_file=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_json_file)
    loaded_model.load_weights(weight_path)
    print("model loaded")
    return loaded_model


def save_model_only(model,string):
    model_json=model.to_json()
    with open(os.path.join(config.output_path(),string),"w") as json_file:
        json_file.write(model_json)
        print("model saved")

#********************************************** Callbacks**************

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


#for early stopping if no improvement within patience batches occured


def set_early_stopping():
    return EarlyStopping(monitor="val_loss",
                         patience=5,
                         mode="auto",
                         verbose=2)
def set_model_checkpoint():
    return ModelCheckpoint(os.path.join(config.weight_path(),MODEL_GAP_ADAM_DA_ALL_WEIGHT),
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 2)





def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss',
                             factor = 0.1,
                             patience = 4,
                            min_lr = 1e-6)









        



















        
    
