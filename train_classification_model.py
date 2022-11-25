import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import efficientnet.tfkeras as efn 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile



def createModel(modelName,saveModelJSON=False):
    """[summary]
    This is a function that will create a model from given modelname and attach final layers according to our need
    """
    if modelName == "VGG16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = VGG16(weights='imagenet', include_top=False)
    elif modelName == "VGG19":
        from tensorflow.keras.applications.vgg19 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = VGG19(weights='imagenet', include_top=False)
    elif modelName == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif modelName == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif modelName == "Xception":
        from tensorflow.keras.applications.xception import preprocess_input
        preprocessing_function = preprocess_input
        base_model = Xception(weights='imagenet', include_top=False)
    elif modelName == "InceptionResNetV2":
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    elif modelName == "MobileNet":
        from tensorflow.keras.applications.mobilenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = MobileNet(weights='imagenet', include_top=False)
    elif modelName == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = MobileNet(weights='imagenet', include_top=False)
    elif modelName == "DenseNet121":
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet121(weights='imagenet', include_top=False)
    elif modelName == "DenseNet169":
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet169(weights='imagenet', include_top=False)
    elif modelName == "DenseNet201":
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet201(weights='imagenet', include_top=False)
    elif modelName == "NASNetLarge":
        from tensorflow.keras.applications.nasnet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = NASNetLarge(weights='imagenet', include_top=False)
    elif modelName == "NASNetMobile":
        from tensorflow.keras.applications.nasnet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = NASNetMobile(weights='imagenet', include_top=False)
    elif modelName == "efficientnet":
        from efficientnet.tfkeras import preprocess_input
        preprocessing_function = preprocess_input
        base_model  = efn.EfficientNetB0(weights="imagenet", include_top=False)
    else:
        ValueError("The model you requested is not supported in Keras")

    preprocess_input=preprocessing_function
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(4096,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(1,activation='sigmoid')(x) #final layer with sigmoid activation
    model=Model(inputs=base_model.input,outputs=preds)
    # un-freeze the BatchNorm layers
    for layer in base_model.layers:
      if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True
    # Unfreeze last 20 layers 
    for layer in model.layers[:20]:
        layer.trainable=True
    
    if saveModelJSON==True:
        model_json = model.to_json()
        filename=str(modelName)+'.json'
        with open(filename, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
    try:
        model.load_weights(modelName+".h5")
    except:
        pass

    return model,preprocess_input

def train(modelName,batch_size,nb_epochs,saveModelJSON=False):
    """[summary]
    This is a function that will create a model according to given arguments and train the classifier
    """

    model,preprocess_input=createModel(modelName,saveModelJSON)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    train_data_dir=os.path.join(os.getcwd(), "train/")
    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.10,rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.01,
                                    zoom_range=[0.9, 1.25],
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    fill_mode='reflect',
                                    data_format='channels_last',
                                    brightness_range=[0.5, 1.5]) 



    train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(224,224), color_mode='rgb',batch_size=batch_size,class_mode='binary',shuffle=True, subset='training') # set as training data
    validation_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(224,224), color_mode='rgb',batch_size=batch_size,class_mode='binary',shuffle=True, subset='validation')
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint_callback = ModelCheckpoint(str(modelName)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    history=model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = nb_epochs,callbacks=[checkpoint_callback,tensorboard_callback])

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(modelName+'_accuracy.png')
    plt.close()
    #plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(modelName+'_Loss_History.png')
    plt.close()
    


# Define Main Function
if __name__ == "__main__":
    args=sys.argv
    myargument=sys.argv
    if len(myargument)==5:

        modelName=str (myargument[2])
        batch_size=int (myargument[3])
        nb_epochs=int (myargument[4])
        train(modelName,batch_size,nb_epochs,saveModelJSON=True)

    else:
        modelName_list=['MobileNet']
        
        batch_size=8
        nb_epochs=20
        for modelName in modelName_list:
            train(modelName,batch_size,nb_epochs,saveModelJSON=True)
        