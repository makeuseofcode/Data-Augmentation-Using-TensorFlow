import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   brightness_range=[0.2,1.0],
                                   zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(directory='/content/drive/MyDrive/cats_and_dogs_filtered/train',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

validation_data = validation_datagen.flow_from_directory(directory='/content/drive/MyDrive/cats_and_dogs_filtered/validation',
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')

test_data = test_datagen.flow_from_directory(directory='/content/drive/MyDrive/cats_and_dogs_filtered/test',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_data,
                    steps_per_epoch=train_data.n // train_data.batch_size,
                    epochs=50,
                    validation_data=validation_data,
                    validation_steps=validation_data.n // validation_data.batch_size)

test_loss, test_acc = model.evaluate(test_data,
                                     steps=test_data.n // test_data.batch_size)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
