import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import tensorflow_datasets as tfds

# Load EMNIST Letters dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    image = tf.image.resize(image, [28, 28])  # Ensure size is 28x28
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    label = tf.cast(label, tf.int32) - 1  # Labels are 1-26, shift to 0-25
    return image, label

ds_train = ds_train.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Define MLP model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(26, activation='softmax')  # 26 classes for letters A-Z
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(ds_train, epochs=10, validation_data=ds_test)

# Evaluate the model
loss, accuracy = model.evaluate(ds_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('handwritten_model.h5')
