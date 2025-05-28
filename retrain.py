import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras import layers, models, optimizers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Preprocessing Functions ---
def load_and_preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = preprocess_input(image)
    return image

def load_data(image_dir):
    dataset = []
    labels = []
    label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
    for label_name in label_map.keys():
        folder = os.path.join(image_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for image_name in os.listdir(folder):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder, image_name)
                image = load_and_preprocess_image(image_path, INPUT_SIZE)
                dataset.append(image)
                labels.append(label_map[label_name])
    return np.array(dataset), np.array(labels)

# --- Load Dataset ---
image_directory = '../dataset/'  # Update path if needed
INPUT_SIZE = 224
dataset, labels = load_data(image_directory)

# --- Split Dataset ---
train_images, test_images, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
val_datagen = ImageDataGenerator()
train_images_aug = train_datagen.flow(train_images, train_labels, batch_size=32)
val_images_aug = val_datagen.flow(val_images, val_labels, batch_size=32)

# --- Class Weights ---
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# --- Build CNN + LSTM Model ---
base_model = DenseNet121(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False, weights='imagenet')
for layer in base_model.layers[:-30]:
    layer.trainable = False  # unfreeze last 30 layers for fine-tuning

inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
x = base_model(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Reshape((1, x.shape[-1]))(x)
x = layers.LSTM(128, return_sequences=False)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = Model(inputs, outputs)

# --- Compile Model ---
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Callbacks ---
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-6),
    ModelCheckpoint("retrained_graph.pb", save_best_only=True, monitor='val_accuracy')
]

# --- Train Model ---
history = model.fit(train_images_aug,
                    epochs=35,
                    validation_data=val_images_aug,
                    class_weight=class_weights,
                    callbacks=callbacks)

# --- Evaluate Model ---
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# --- Plot Accuracy and Loss ---
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# --- Confusion Matrix ---
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3', '4'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
