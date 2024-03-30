#Imported libraries
import tensorflow as tf 
import numpy as np 
import os

# Paths 
path = '/Users/whybless/ai/computer_vision/image_classification'
base_path = os.path.join(path, 'locations_dataset.zip')

# Training pictures
train_buildings = os.path.join(base_path,'seg_train/buildings')
train_forest = os.path.join(base_path,'seg_train/forest')
train_glacier = os.path.join(base_path,'seg_train/glacier')
train_mountain = os.path.join(base_path,'seg_train/mountain')
train_sea = os.path.join(base_path,'seg_train/sea')
train_street = os.path.join(base_path,'seg_train/street')

#validation pictures
validation_buildings = os.path.join(base_path, 'seg_pred/buildings')
validation_forest = os.path.join(base_path, 'seg_pred/forest')
validation_glacier = os.path.join(base_path, 'seg_pred/glacier')
validation_mountain = os.path.join(base_path, 'seg_pred/mountain')
validation_sea = os.path.join(base_path, 'seg_pred/sea')
validation_street = os.path.join(base_path, 'seg_pred/street')

# Test pictures 
test_buildings = os.path.join(base_path, 'seg_test/buildings')
test_forest = os.path.join(base_path, 'seg_test/forest')
test_glacier = os.path.join(base_path, 'seg_test/glacier')
test_mountain = os.path.join(base_path, 'seg_test/mountain')
test_sea = os.path.join(base_path, 'seg_test/sea')
test_street = os.path.join(base_path, 'seg_test/street')


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preprocessing using ImageDataGenerator
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    base_path + '/seg_train',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    base_path + '/seg_pred',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory




# Labelling


#Model Selection

import tensorflow as tf
from tensorflow import keras

from keras import layers, models

# Load pre-trained ResNet model
resnet_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers
for layer in resnet_model.layers:
    layer.trainable = False

# Define model
model = models.Sequential()

# Add the ResNet model as a layer
model.add(resnet_model)

# Add additional layers on top of ResNet
model.add(layers.Flatten())  # Example: Flatten layer
model.add(layers.Dense(256, activation='relu'))  # Example: Dense layer with 256 units and ReLU activation
model.add(Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))  # Example: Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()








#Early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training
epochs = 20
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,  # Specify the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('image_classification_model.h5')

#Validation and Hyperparameter Tuning
import matplotlib.pyplot as plt

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Testing

trained_model = keras.models.load_model('image_classification_model.h5')

# Create a new test generator
test_generator = test_datagen.flow_from_directory(
    base_path + '/seg_test',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# Evaluate the model on the test set
test_loss, test_accuracy = trained_model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy}')


