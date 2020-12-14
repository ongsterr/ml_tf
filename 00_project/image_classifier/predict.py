import numpy as np
import matplotlib.pyplot as plt 

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

def get_parameters():
  parser = argparse.ArgumentParser()

  parser.add_argument('image', action='store', metavar='image', help='Specify filepath for image file')
  parser.add_argument('model', action='store', metavar='model', help='Specify filepath for model.h5 file')
  
  parser.add_argument('-k', '--top_k', action='store', dest='top_k', type=int, help='Specify top k most likely classes')
  parser.add_argument('-c', '--category_names', action='store', dest='classes_filepath', help='Specify filepath to json file for category names')

  results = parser.parse_args()

  return results.image, results.model, results.top_k, results.classes_filepath

def get_class_names(json_filepath):
  class_names = []

  if json_filepath != None:
    with open(json_filepath, 'r') as f:
      class_names_dict = json.load(f)

    for n in range(102):
      class_names.append(class_names_dict['{}'.format(n+1)])
    
  return class_names

def process_image(img_arr, img_size=224):
  image = tf.cast(img_arr, tf.float32)
  image = tf.image.resize(image, (img_size, img_size))
  image /= 255.

  return image

def predict(image_path, model, top_k, class_names=[]):
  img = Image.open(image_path)
  img = np.asarray(img)
  img = process_image(img)
  img = np.expand_dims(img, 0)
  pred = model.predict(img)

  top_k_classes = np.argpartition(pred[0], -top_k)[-top_k:]
  top_k_preds = np.partition(pred[0], -top_k)[-top_k:]

  if len(class_names) == len(np.arange(102)):
    top_k_classnames = [class_names[x] for x in top_k_classes]
  else:
    top_k_classnames = (top_k_classes + 1).astype('str')

  return top_k_preds, top_k_classnames, np.squeeze(img, axis=0)

def plot_prob_distribution(probs, classes, image):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  ax1.imshow(image, cmap=plt.cm.binary)
  ax1.axis('off')

  ax2.barh(classes, probs)
  ax2.set_yticks(classes)
  ax2.set_yticklabels(classes)
  ax2.set_title('Class Probability')
  ax2.set_xlim(0, 1.1)
  plt.tight_layout()
  plt.show()

def run():
  image_filepath, model_filepath, top_k, classes_filepath = get_parameters()

  class_names = get_class_names(classes_filepath)
  loaded_model = tf.keras.models.load_model(model_filepath, custom_objects={'KerasLayer': hub.KerasLayer})

  top_k_classes = top_k if top_k != None and int(top_k or 0) > 0 else 5
  probs, classes, image = predict(image_filepath, loaded_model, top_k_classes, class_names)

  plot_prob_distribution(probs, classes, image)

run()