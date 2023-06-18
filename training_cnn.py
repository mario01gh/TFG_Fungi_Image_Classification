import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,load_model
from tensorflow.keras.utils import  set_random_seed
from keras.layers import Dense, Dropout, BatchNormalization,Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator  
import splitfolders
import os
import sys

# Obtencion de argumentos
# --------------------------------------------------------------------------------------------------------
# python training_cnn.py model_name input_size batch_size learning_rate epochs cnn_architecture filter_list kernel_list FC_list dropout_list batch_norm
print(sys.argv)
if len(sys.argv) == 12:
        model_name = sys.argv[1]
        width = int(sys.argv[2])
        height = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        lr = float(sys.argv[4])
        n_epochs = int(sys.argv[5])
        cnn_arch = sys.argv[6]
        filter_list = list(map(int, sys.argv[7].strip('[]').split(',')))
        kernel_list = list(map(int, sys.argv[8].strip('[]').split(',')))
        FC_list = list(map(int, sys.argv[9].strip('[]').split(',')))
        dropout_list = list(map(float, sys.argv[10].strip('[]').split(',')))
        batch_norm = sys.argv[11]
else:
    print("Error, numero incorrrecto de argumentos")
    print("Ejemplo:python training_cnn.py model_1 224 32 0.0001 10 cnn_1 [64,64,128,128] [3,3,5,5] [128,128] [0.5] no")

# --------------------------------------------------------------------------------------------------------


# Cargar dataset
# --------------------------------------------------------------------------------------------------------

dataset_path = # escribir aqui la ruta del dataset

# modificar generadores para incluir data augmentation

train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=40,
                                zoom_range=0.3,
                                width_shift_range=0.4,
                                height_shift_range=0.4,
                                shear_range=0,
                                horizontal_flip=True,
                                vertical_flip=False,
                                fill_mode="nearest")
                        
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        dataset_path+'/train',
        target_size=(width, height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        seed = 202)
        
validation_generator = test_datagen.flow_from_directory(
        dataset_path+'/val',
        target_size=(width, height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        seed = 202)

test_generator = test_datagen.flow_from_directory(
        dataset_path+'/test',
        target_size=(width, height),
        batch_size=batch_size,
        shuffle=False,
        class_mode="categorical")

# --------------------------------------------------------------------------------------------------------


# Creacion, compilación y ejecución del modelo
# --------------------------------------------------------------------------------------------------------

parent_dir = # escribir aqui el directorio de destino de los modelos
path = os.path.join(parent_dir, model_name)
if not os.path.exists(path):
        os.mkdir(path)
results_path = parent_dir+"/"+model_name+"/"

checkpoint = ModelCheckpoint(results_path+"best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)

# Se establece una semilla para asegurar la reproduccion de resultados
semilla = 2023
set_random_seed(semilla)

num_classes = 20

model_cnn= Sequential()
i = 0
while(i < len(filter_list)):
        kernel_size = (kernel_list[i], kernel_list[i])
        if i == 0:
                model_cnn.add(Conv2D(filter_list[i],kernel_size,input_shape=(width,height,3), activation='relu'))
        else: 
                model_cnn.add(Conv2D(filter_list[i],kernel_size, activation='relu'))

        if cnn_arch == "cnn_2":
                i+=1
                kernel_size = (kernel_list[i], kernel_list[i])
                model_cnn.add(Conv2D(filter_list[i],kernel_size, activation='relu'))
        
        if batch_norm == "si":
                model_cnn.add(BatchNormalization())

        model_cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model_cnn.add(Dropout(dropout_list[0]))
        i+=1

model_cnn.add(Flatten())
for index in range(len(FC_list)):
        model_cnn.add(Dense(FC_list[index], activation='relu'))
        if len(dropout_list) == 1:
                model_cnn.add(Dropout(dropout_list[0]))
        else:
                model_cnn.add(Dropout(dropout_list[1]))

model_cnn.add(Dense(num_classes, activation='softmax'))
print(model_name)
model_cnn.summary()

model_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'])

# avoid warning and info messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# solving out of memory errors
config =tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

historico_cnn = model_cnn.fit(
        train_generator,
        epochs=n_epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint]
        )

# --------------------------------------------------------------------------------------------------------


# Almacenamiento de graficas, resultados y matrices de confusion
# --------------------------------------------------------------------------------------------------------

#Gráfica del loss del modelo
plt.plot(historico_cnn.history['loss'])
plt.plot(historico_cnn.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(results_path+"loss_graph.png")
plt.close()

#Gráfica de accuracy del modelo
plt.plot(historico_cnn.history['categorical_accuracy'])
plt.plot(historico_cnn.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.ylim([0, 1])
plt.savefig(results_path+"accuracy_graph.png")
plt.close()

# Se imprimen por terminal valores de loss y accuracy para todos los conjuntos de datos
ciclo_final=np.where(historico_cnn.history['val_loss'] == np.min(historico_cnn.history['val_loss']))
ciclo_final = ciclo_final[0][0]

train_loss = historico_cnn.history['loss'][ciclo_final]
train_acc = historico_cnn.history['categorical_accuracy'][ciclo_final]

val_loss = historico_cnn.history['val_loss'][ciclo_final]
val_acc = historico_cnn.history['val_categorical_accuracy'][ciclo_final]

print("Mejor ciclo: ", ciclo_final)
print('Training loss:', train_loss)
print('Training categorical accuracy:', train_acc)
print('Validation loss:', val_loss)
print('Validation categorical accuracy:', val_acc)

# Se carga el archivo almacenado por el metodo de checkpoint
saved_model = load_model(results_path+'best_model.h5')

evaluacion_cnn = saved_model.evaluate(test_generator)
test_loss = evaluacion_cnn[0]
test_acc = evaluacion_cnn[1]
print(f'Test results - Loss: {test_loss} - Accuracy: {test_acc}')

# Almacenamiento de los resultados en un archivo csv
resultados_csv = # escribir aqui la ruta del archivo csv 
with open(resultados_csv, "a") as f1:
        f1.write(model_name+","+str(train_loss)+","+str(train_acc)+","+str(val_loss)+","+str(val_acc)+","+str(test_loss)+","+str(test_acc)+","+str(ciclo_final)+"\n")
f1.close()

# obtenemos las predicciones realizadas por el modelo 
test_pred = saved_model.predict(test_generator)
class_pred =  np.argmax(test_pred, axis=1)

# se calcula la matriz de confusion
conf_matrix = np.transpose(confusion_matrix(test_generator.labels, class_pred))
print(conf_matrix)

class_names = list(test_generator.class_indices.keys())
report = classification_report(test_generator.labels, class_pred, target_names=class_names)
print(report)

# almacenamos una imagen de la matriz de confusion
labels = list(test_generator.class_indices.keys())
ig, axis = plt.subplots(figsize=(8, 8))
im = axis.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)

axis.set(xticks=np.arange(conf_matrix.shape[1]),
       yticks=np.arange(conf_matrix.shape[0]),
       xticklabels=labels, yticklabels=labels,
       xlabel='Real', ylabel='Predicción')
    
axis.set_xticklabels(labels, rotation=90, fontsize=10)
axis.set_yticklabels(labels, fontsize=10)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        axis.text(j, i, format(conf_matrix[i, j], 'd'),ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
plt.savefig(results_path+'confusion_matrix.png', bbox_inches='tight')

# se guarda el classification report
with open(results_path+'classification_report.txt', 'w') as f:
    f.write(report)
f.close()
# se guarda la matriz de confusion
np.savetxt(results_path +'matrizConf_cnn.txt', conf_matrix, fmt='%-3d')
# se guardan los pesos
saved_model.save_weights(results_path +'pesos_cnn.h5')

# --------------------------------------------------------------------------------------------------------