# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:07:09 2020

@author: HOME PC
"""


#### Importe de Librerias a Usar
import pydot
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

######### PreProcesamiento

# Carga un archivo a la memoria
def load_doc(filename):
    # Abre el archivo en solo lectura
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
# Carga las imagenes con sus etiquetas
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions ={}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions: 
            descriptions[img[:-2]]=[caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions
#Limpieza de las etiquetas, puntuación, convertir a minusculas y demás
def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):
            img_caption.replace("-"," ")
            desc = img_caption.split()
            #Convierte a minuscula
            desc = [word.lower() for word in desc]
            #Quita los signos de puntuacion
            desc = [word.translate(table) for word in desc]
            #Quita apostofres de las palabras
            desc = [word for word in desc if(len(word)>1)]
            #Quita las palabras que tengan números 
            desc = [word for word in desc if(word.isalpha())]
            #Se convierte a tipo string
            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions
def text_vocabulary(descriptions):
    # Armar vocabulario de palabras 
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab
#Todas las descripciones en un archivo
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list: 
             lines.append(key + '\t' + desc )
             data = "\n".join(lines)
    file = open(filename,"w")
    file.write(data)
    file.close()
    
##### Definir paths donde se encuentran los datos    
dataset_text = "F:/Unal/2020-1/Mineria/Proyecto/Flickr8k_text"
dataset_images = "F:/Unal/2020-1/Mineria/Proyecto/Flicker8k_Dataset"
filename = dataset_text + "/" + "Flickr8k.token.txt"


### Creación archivo descripciones.
#Cargando archivos que tienen las descripciones
#Creando diccionario que tiene las 5 descripciones por cada imagen
descriptions = all_img_captions(filename)
print("Length of descriptions =" ,len(descriptions))
#limpieza de las descripciones
clean_descriptions = cleaning_text(descriptions)
#armando vocabulario
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#Guardando las descripciones en un archivo 
save_descriptions(clean_descriptions, "descriptions.txt")
descriptions


### Preprocesamiento Xception
def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            features[img] = feature
        return features
#Longitud del vector de salida : 2048
#features = extract_features(dataset_images)


features = load(open("L:/Mineria/features.p","rb"))

#Carga de datos 
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos
def load_clean_descriptions(filename, photos): 
    #Cargar descripciones limpias
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions
def load_features(photos):
    #Cargando preprocesamiento
    all_features = load(open("F:/Unal/2020-1/Mineria/features.p","rb"))
    #Se seleccionan los vectores de preprocesamiento que seran solamente usados en train
    features = {k:all_features[k] for k in photos}
    return features

#### Generación de datos de training.
filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)


#Convierte el diccionario en una lista
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
#Crea la clase tokenizer
#Cada número correspondera a una de las palabras.
from tensorflow.keras.preprocessing.text import Tokenizer
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer
#Crea el tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
vocab_size
tokenizer.word_index

#Calculo de la longitud maxima de la descripcion
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(descriptions)
max_length


#### Creación data-generator
#Crea secuencias de las descripciones
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield [[input_image, input_sequence], output_word]
def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    #se hace para las 5 descripciones de la imagen
    for desc in desc_list:
        # se transforma la descripcion en números segun tokenizer
        seq = tokenizer.texts_to_sequences([desc])[0]
        # separa una descripcion por palabras
        for i in range(1, len(seq)):
            # separacion entre los de entrada y salida
            in_seq, out_seq = seq[:i], seq[i]
            # secuencia de entrada
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # transformacion de la salida
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # almacena
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

#### Modelo
    
def define_model(vocab_size, max_length):
    #Modelo CNN
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    #Modelo LSTM
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    #Juntar los dos modelos
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)  
    return model
#Modelo

model = define_model(vocab_size, max_length)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
print(os.environ["PATH"])

epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    history=model.fit_generator(generator, epochs=epochs, steps_per_epoch= steps, verbose=1)
    
def extract_features(filename, model):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
def word_for_id(integer, tokenizer):
   for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
         return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


img_path = ("F:/Unal/2020-1/Mineria/Proyecto/test/166321294_4a5e68535f.jpg")
img = Image.open("F:/Unal/2020-1/Mineria/Proyecto/test/166321294_4a5e68535f.jpg")
photo = extract_features(img_path, xception_model)


description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)    
    