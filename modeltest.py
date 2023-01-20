import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
#from tqdm.notebook import tqdm
import warnings, cv2
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

train_dir = "dataset_new\\train"
test_dir = "dataset_new\\test"
print(train_dir)
#dataset_new\train

def load_images(directory):
    images = []
    labels = []
    
    for category in os.listdir(directory):
        for filename in (os.listdir(directory+"\\" +category)):
            image_path = os.path.join(directory,category,filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(224,224))
            images.append(image)
            labels.append(category)
    
    images = np.array(images,dtype='float32')
    return images, labels

X_train, y_train = load_images(train_dir)
X_test, y_test = load_images(test_dir)

X_train = X_train / 255.
X_test = X_test / 255.
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn = Sequential()

cnn.add(Conv2D(filters=16,kernel_size=3,activation='relu',input_shape=(224,224,3)))
cnn.add(MaxPooling2D(pool_size=2))

cnn.add(Conv2D(filters=32,kernel_size=3,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.2))

cnn.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(units=128,activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))

cnn.add(Dense(units=4,activation='softmax'))
cnn.summary()
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy',patience=20,mode='max',verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',patience=5,mode='max',verbose=1,factor=0.1,min_lr=0.001)
checkpoint_filename = 'checkpoint/'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filename,monitor='val_accuracy',verbose=1,save_best_only=True,save_weights_only=True,mode='max')

k = cnn.fit(x=X_train,
            y=y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test,y_test),
            callbacks=[early_stopping,reduce_lr,model_checkpoint])

cnn.load_weights(checkpoint_filename)

plt.plot(k.history['accuracy'], label='train acc')
plt.plot(k.history['val_accuracy'], label='val acc')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.show()
plt.savefig('AccVal_acc.jpg')

plt.figure(figsize=(12,8))
plt.plot(k.history['loss'],'r',label='train loss')
plt.plot(k.history['val_loss'],'b',label='test loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()
plt.savefig('LossVal_loss.jpg')

y_pred = cnn.predict(X_test)

#y_pred=np.argmax(y_pred, axis=1)
#y_test=np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
plt.savefig("Cnf matrix.jpg")
print("Recall ", recall_score(y_test,y_pred,average = None))
print("Precision ", precision_score(y_test,y_pred,average = None))
print("F1 score ", f1_score(y_test,y_pred,average = None))
print("Accuracy score ", accuracy_score(y_test,y_pred))
plot_model(cnn, to_file='model_plot.png', show_shapes=True, show_layer_names=True , show_layer_activations=True)

#cnn.save('CNNclass_1_20.h5')
