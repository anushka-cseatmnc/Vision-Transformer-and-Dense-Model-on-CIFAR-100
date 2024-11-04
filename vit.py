# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load and Preprocess CIFAR-100 Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize images

# Reshape images for ViT (adding channel dimension if not present)
train_images_vit = np.expand_dims(train_images, -1)
test_images_vit = np.expand_dims(test_images, -1)

# Step 3: Define the Basic Dense Model
basic_model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),  # Flatten input for CIFAR-100 shape
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(100, activation='softmax')  # Output layer for 100 classes
])

# Step 4: Compile the Basic Model
basic_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Step 5: Train the Basic Model
basic_model.fit(train_images, train_labels, epochs=10)

# Step 6: Evaluate the Basic Model
test_loss, test_accuracy = basic_model.evaluate(test_images, test_labels)
print(f"Basic Model Test accuracy: {test_accuracy}")

# Step 7: Define the Vision Transformer Model for CIFAR-100
class VisionTransformer(tf.keras.Model):
    def __init__(self, num_patches, projection_dim, num_heads, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = layers.Dense(projection_dim)
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.layer_norm = layers.LayerNormalization()
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.flatten_layer = layers.Flatten()
        self.dense_mlp = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, patches):
        patches = self.patch_embedding(patches)
        attention_output = self.attention_layer(patches, patches)
        x = self.layer_norm(attention_output)
        x = self.global_average_pooling(x)
        x = self.flatten_layer(x)
        return self.dense_mlp(x)

# Step 8: Prepare Data Patches for Vision Transformer
patch_size = 8  # Smaller patch size due to higher resolution
num_patches = (32 // patch_size) ** 2
projection_dim = 64
num_heads = 4
num_classes = 100

def create_patches(images, patch_size):
    batch_size, height, width, channels = images.shape
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches

train_patches = create_patches(train_images, patch_size)
test_patches = create_patches(test_images, patch_size)

# Step 9: Initialize and Compile the Vision Transformer Model
vit_model = VisionTransformer(num_patches, projection_dim, num_heads, num_classes)
vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 10: Train the Vision Transformer Model
vit_model.fit(train_patches, train_labels, epochs=10)

# Step 11: Evaluate the Vision Transformer Model
vit_test_loss, vit_test_accuracy = vit_model.evaluate(test_patches, test_labels)
print(f"Vision Transformer Model Test accuracy: {vit_test_accuracy}")

# Step 12: Make Predictions with Both Models
basic_predictions = basic_model.predict(test_images)
vit_predictions = vit_model.predict(test_patches)

# Step 13: Visualize Predictions for Both Models
def plot_predictions(images, predictions, labels, title="Predictions"):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        predicted_label = np.argmax(predictions[i])
        true_label = labels[i][0]
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f'Pred: {predicted_label} (True: {true_label})', color=color)
    plt.suptitle(title)
    plt.show()

# Visualize basic model predictions
plot_predictions(test_images, basic_predictions, test_labels, title="Basic Model Predictions")

# Visualize Vision Transformer model predictions
plot_predictions(test_images, vit_predictions, test_labels, title="Vision Transformer Model Predictions") 
 
