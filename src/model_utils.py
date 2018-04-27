from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,ReduceLROnPlateau
import numpy
import os
from .. import config
print("saiful")

#model name

MAIN_MODEL="my_resnet_inception_v2.json"
MAIN_MODEL_WEIGHT="my_resnet_inception_v2.h5"
def save_model(model):
    """
    save model to output directory
    """
    
    #serialize the model to json
    model_json=model.to_json()
    #write
    with open(os.path.join(config.output_path(),MAIN_MODEL), "w") as json_file:
        json_file.write(model_json)
    print("model saved")

    #save the weight
    #serialize weights to HDF5
    model.save_weights(os.path.join(config.output_path(),MAIN_MODEL_WEIGHT))
    print("weight saved")
def load_model(string):
    """
    load a model from output directory by name
    """



#********************************************** Callbacks**************

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get("loss"))
        sefl.val_losses.append(logs.get("val_loss"))


#for early stopping if no improvement within patience batches occured


def set_early_stopping():
    return EarlyStopping(monitor="val_loss",
                         patience=5,
                         mode="auto",
                         verbose=2)
def set_model_checkpoint():
    return ModelCheckpoint(os.path.join(config.output_path(),MAIN_MODEL_WEIGHT),
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









        



















        
    
