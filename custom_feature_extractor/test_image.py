import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

IMAGE_SIZE = 256

# Create an ImageDataGenerator for a single image
test_datagen_single = ImageDataGenerator(rescale=1/255)

# Load a single image for testing
image_path = 'vehicle_dataset_test/178/S01_c005_119_9.jpg'  # Replace with the actual path to your single image
image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
image_array = img_to_array(image)
image_array = image_array.reshape((1,) + image_array.shape)  # Add batch dimension
image_array /= 255.0  # Normalize pixel values to be between 0 and 1

# Create a flow for the single image using the generator
test_generator_single = test_datagen_single.flow(image_array, batch_size=1)

# Load the model
model = tf.keras.models.load_model('custom_feature_model')

# Print model summary
print(model.summary())

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Predict the class probabilities for the single image
predictions = model.predict(image_array)

# Print the predicted class probabilities
print("Predicted class probabilities:", predictions)

# Get the predicted class (index with the highest probability)
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
print("Predicted class:", predicted_class)
