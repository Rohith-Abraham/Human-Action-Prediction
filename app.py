from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # Import the image module
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Create a data generator for loading images
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Specify the directory where your flowers are stored
train_data_dir = "D:\\sk sir\\lab2\\human_action"  # Update this path

# Create the train generator
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Now you can use train_generator.class_indices.keys() as needed


app = Flask(__name__)

# Define the path for uploads directory
uploads_dir = os.path.join(app.root_path, 'static', 'uploads')

# Create uploads directory if it doesn't exist
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Load the trained model
model = load_model("D:\\sk sir\\lab2\\model.h5")  # Replace with the actual path

# Define the list of flower names manually (based on your model's training)
humanactions_names = ['Clapping', 'Cycling', 'Dancing', 'Drinking', 'Eating','Fighting','Laughing','Running','Sleeping']  # Update this list as needed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # Save the uploaded file to the uploads directory
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Rescale image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Print the shape of the input image
        print("Image shape: ", img_array.shape)

        # Make predictions using the model
        predictions = model.predict(img_array)
        print("Predictions: ", predictions)  # Debugging statement
        
        # Check the model's class indices and compare to the prediction
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class index: ", predicted_class[0])  # Debugging statement
        
        # Ensure flower names are in the same order as the class indices
        predicted_humanactions = humanactions_names[predicted_class[0]]
        
        # Pass the filename and predicted class name to the result template
        return render_template('result.html', actions=predicted_humanactions, filename=file.filename)

    return render_template('index.html')

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)


