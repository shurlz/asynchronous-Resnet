import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import Input, Concatenate, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
import asyncio
import time
from PIL import Image
from encoded_labels import labels

target_size = 331

text_labels = list(labels.keys())
numeric_labels = list(labels.values())

model = tf.keras.models.load_model('final_dog_classifier.h5')

async def get_features(MODEL, generator, preprocess_input):
    model = MODEL(include_top=False, weights='imagenet', pooling='avg')
    inputs = Input((331, 331, 3))
    x = Lambda(preprocess_input, name='preprocessing')(inputs)
    x = (model)(x)
    model = Model(inputs, x)

    features = model.predict(generator)
    return features


async def breed_checker(img_path):
    image_reshaped = image.load_img(img_path, target_size=(target_size, target_size), color_mode='rgb')
    img_array = image.img_to_array(image_reshaped)
    img_batch = np.expand_dims(img_array, axis=0)  # makes it 4 dimensional

    parallel_extraction = await asyncio.gather(get_features(ResNet50, img_batch, resnet_preprocess),
                                               get_features(Xception, img_batch, xcep_preprocess))
    resnet_features = parallel_extraction[0]
    xception_features = parallel_extraction[1]

    features = np.concatenate([xception_features, resnet_features], axis=-1)

    prediction = model.predict(features)  # predicting

    highest = pd.DataFrame([x * 100 for x in prediction[0]], text_labels).sort_values(0, ascending=False)[
              :5]  # map probability to breed

    return highest

start = time.perf_counter()
asyncio.run(breed_checker('retriever.png'))
end = time.perf_counter()
print(f'took {end - start} seconds to execute')
