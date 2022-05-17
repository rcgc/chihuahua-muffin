import os
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers


base_dir = "chihuahua_vs_muffin"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Loading datasets into tensors
train_dataset = image_dataset_from_directory(
    directory=train_dir,
    image_size=(112, 112),
    batch_size=10)

validation_dataset = image_dataset_from_directory(
    directory=validation_dir,
    image_size=(112, 112),
    batch_size=10)

test_dataset = image_dataset_from_directory(
    directory=test_dir,
    image_size=(112, 112),
    batch_size=10)

# Data augmentation generation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(112, 112),
    batch_size=10,
    class_mode="binary")

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(112, 112),
    batch_size=10,
    class_mode="binary")

# Setup network
conv_base = VGG19(include_top=False,
                  weights="imagenet",
                  input_shape=(112, 112, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1000))
model.add(layers.Dense(128, activation="softmax"))
model.add(layers.Dense(1, activation="sigmoid"))

print(model.summary())

conv_base.trainable = False

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adagrad(),
    metrics=['acc']
)

n_training_images = 100
n_validation_images = 50
batch_size = 10

n_steps_epoch = n_training_images / batch_size
n_validation_steps = n_validation_images / batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=n_steps_epoch,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=n_validation_steps)

model.save("vgg19_chihuahua_vs_muffin.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'b', label='train accuracy')
plt.plot(epochs, val_acc, 'orange', label='validation accuracy')
plt.title('train acc vs val acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='training loss')
plt.plot(epochs, val_loss, 'orange', label='validation loss')
plt.title('train loss vs val loss')
plt.legend()

# plt.show(block=False)

test_datagen = ImageDataGenerator(1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(112, 112),
    batch_size=25,
    class_mode="binary")

test_loss, test_acc = model.evaluate(test_generator, steps=25)
print("\ntest acc :\n", test_acc)

plt.show()
