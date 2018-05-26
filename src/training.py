""" train a inception network using eyepack dataset"""

# python packages
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD,RMSprop,Adam,Adagrad

from keras.optimizers import SGD

# project modules

from . import models
from .. import config
from . import model_utils


#calculation for train data
total_train_images = config.total_count_files(config.TRAIN_DIR)
steps_train = int(total_train_images // config.batch_size)


#calculation for validation data
total_val_images = config.total_count_files(config.VAL_DIR)
steps_val = int(total_val_images // config.batch_size)


#datagenerator object

train_datagen=ImageDataGenerator(rescale=1/255,

                                rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest'



                                 )

train_datagen = ImageDataGenerator(rescale = 1/255)



#train data generator
train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    classes = ["0","1","2","3","4"],
    target_size = config.image_size_gen,
    batch_size = config.batch_size,
    class_mode = 'categorical')



#validation data generator
val_generator = train_datagen.flow_from_directory(
    config.VAL_DIR,
    classes = ["0","1","2","3","4"],
    target_size = config.image_size_gen,
    batch_size = config.batch_size,
    class_mode = 'categorical')




model=models.custom_inceptionResnetV2_conv_global()

#optimizer
optimizer=SGD(lr=1e-3,
              momentum=0.9,
              nesterov=True)
optimizer_rms=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)

optimizer_adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

optimizer_adagrad=Adagrad(lr=0.001, epsilon=None, decay=0.0)
    
objective="categorical_crossentropy"

model.compile(optimizer=optimizer_adam,
              loss=objective,
              metrics=['accuracy']
              )

# constructing model
model = models.custom_inceptionResnetV2()

# optimizer
optimizer=SGD(lr = 1e-3,
              momentum = 0.9,
              nesterov = True)

objective="categorical_crossentropy"

model.compile(optimizer = optimizer,
              loss = objective,
              metrics = ['accuracy'])




# callbacks
history = model_utils.LossHistory()
early_stopping = model_utils.set_early_stopping()
model_cp = model_utils.set_model_checkpoint()
reduce_lr = model_utils.set_reduce_lr()




# training model
history = model.fit_generator(train_generator,
                            steps_per_epoch = steps_train,
                            epochs = config.epochs,
                            callbacks=[history, early_stopping, model_cp, reduce_lr],
                            validation_data = val_generator,
                            validation_steps = steps_val,
                            verbose = 2)


#training
history=model.fit_generator(train_generator,
                            steps_per_epoch=steps_train,
                            verbose=2,
                            epochs=config.epochs,
                            callbacks=[history,early_stopping,model_cp,reduce_lr],
                            validation_data=val_generator,
                            validation_steps=steps_val)

                         
model_utils.save_model(model,model_utils.MODEL_CONV,model_utils.MODEL_GAP_ADAM_DA_ALL_WEIGHT)


# saving model                         
model_utils.save_model(model)











