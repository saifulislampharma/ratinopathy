from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from . import models
from .. import config
from . import model_utils

# test
#calculation for train data
total_train_images=config.total_count_files(config.TRAIN_DIR)
steps_train=int(total_train_images//config.batch_size)

#calculation for validation data
total_val_images=config.total_count_files(config.VAL_DIR)
steps_val=int(total_val_images//config.batch_size)

#datagenerator object
train_datagen=ImageDataGenerator(rescale=1/255)

#train data generator
train_generator=train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    classes=["0","1","2","3","4"],
    target_size=config.image_size_gen,
    batch_size=config.batch_size,
    class_mode='categorical'

    )

#validation data generator
val_generator=train_datagen.flow_from_directory(
    config.VAL_DIR,
    classes=["0","1","2","3","4"],
    target_size=config.image_size_gen,
    batch_size=config.batch_size,
    class_mode='categorical'

    )
##for batch_d,batch_l in train_generator:
##    print("data batch shape",batch_d.shape)
##    print("data label shape",batch_l.shape)
##    break

model=models.custom_inceptionResnetV2()

#optimizer
optimizer=SGD(lr=1e-3,
              momentum=0.9,
              nesterov=True

    )
objective="categorical_crossentropy"

model.compile(optimizer=optimizer,
              loss=objective,
              metrics=['accuracy']
              )

# Training and Evaluating 
history = model_utils.LossHistory()
early_stopping = model_utils.set_early_stopping()
model_cp = model_utils.set_model_checkpoint()
reduce_lr = model_utils.set_reduce_lr()





#training
history=model.fit_generator(train_generator,
                            steps_per_epoch=steps_train,
                            epochs=config.epochs,
                            callbacks=[history,early_stopping,model_cp,reduce_lr],
                            validation_data=val_generator,
                            validation_steps=steps_val)

                         
model_utils.save_model(model)

