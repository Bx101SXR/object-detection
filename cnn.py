import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (100, 100)
batch_size = 32
epochs = 100

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.8)

train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
model.save('object_classifier.h5')

with open("class_labels.txt", "w") as f:
    for name, index in train_generator.class_indices.items():
        f.write(f"{index},{name}\n")
