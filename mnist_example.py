import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_explain.core.integrated_gradients import IntegratedGradients
import matplotlib.pyplot as plt

# ---- Using tf-explain: Integrated Gradients ----

# Load model
model = keras.models.load_model('mnist_model.h5')

# Load MNIST dataset again
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values (very important to match training)
x_test = x_test / 255.0

# Select one image and label to explain
image = x_test[0]       # shape (28, 28)
label = int(y_test[0])

# Expand dims: add batch dimension
image_exp = np.expand_dims(image, axis=0)  # (1, 28, 28)
image_exp = np.expand_dims(image_exp, axis=-1)  # shape (1, 28, 28, 1)

# Create explainer object
explainer = IntegratedGradients()

# Generate explanation; pass None as label to explain predicted class
explanation = explainer.explain(
    validation_data=(image_exp, None),
    model=model,
    class_index=label
)

# Save explanation heatmap
explainer.save(explanation, '.', 'integrated_gradients.png')

heatmap = np.squeeze(explanation)  # shape now (28, 28)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Original grayscale image
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

# Just the heatmap
axs[1].imshow(heatmap, cmap='jet')
axs[1].set_title("Integrated Gradients")
axs[1].axis('off')

# Overlay of both
axs[2].imshow(image, cmap='gray')
axs[2].imshow(heatmap, cmap='jet', alpha=0.5)
axs[2].set_title("Overlay")
axs[2].axis('off')

plt.tight_layout()
plt.show()
