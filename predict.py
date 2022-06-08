import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import argparse
import json
# Ignore everthing from tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# SET IMAGE SIZE (224x224)
IMAGE_SIZE = 224

# Instantiate Argument Parser
parser = argparse.ArgumentParser(
    description='Predict Flower Image',
)

parser.add_argument('image_path', action="store",)
parser.add_argument('model_path',action="store",)
parser.add_argument('--category_names', action="store", dest="category_names_path", default = "label_map.json") 
parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=1)


# Function for loading a saved model 
# model_name: string (name of model)
def load_model(model_name):
    return tf.keras.models.load_model(f"./{model_name}",custom_objects={'KerasLayer':hub.KerasLayer},compile = False)

# Function for processing the input image (formatting it for prediction usage)
# img: image as numpy array
def process_image(img):
    # Cast the type (float tensor)
    tensor = tf.cast(img, tf.float32)
    # Resize the image to given constraints (img_size)
    image = tf.image.resize(tensor, (IMAGE_SIZE, IMAGE_SIZE))
    # Divide by max value (normalize)
    image /= 255
    # return processed image
    return image

# Function for prediction 
# image_path: directory path to image
# model: keras model 
# top_k: number of top classes which should be returned (by default 1)
def predict(image_path, model, top_k, cat_path):
    # Open the image (PIL Image Object)
    img = Image.open(image_path)
    # Convert Image Object to np array
    img_raw = np.asarray(img)
    # Process Image
    img_processed = process_image(img_raw)
    # Expand dimension (add 1 to shape)
    img_expanded = np.expand_dims(img_processed,axis=0)
    # Make a prediction using the model
    prediction = model.predict(img_expanded)
    # Get indices of top_k values
    # using index sorting
    indices = prediction[0].argsort()[-top_k:][::-1]
    # Get the values of the top_k indices
    probs = prediction[0][indices]
    # Open and load the json file
    with open(cat_path, 'r') as f:
        class_names = json.load(f)
    
    # Convert the indices to class_names (change to 1-based labels)
    for i,index in enumerate(indices):
        p = np.round(probs[i],3)
        print(i+1,p,class_names[str(index+1)])

# Parse the arguments
args = parser.parse_args()
# Predict using the predict function and the parsed arguments
predict(args.image_path, load_model(args.model_path), args.top_k, args.category_names_path)
