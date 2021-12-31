import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

print("using TensorFlow ",tf.__version__)
    
diagnostic = False

#like ..
def parent(path):
    #check that path is not /, if so return /
    if len(path) > 1: 
        path = path[:path[:-1].rfind("/")+1]
    return path

#path to the script itself
file_path = os.path.realpath(__file__)
dir_path = parent(file_path)

#check if model has already been trained
if os.path.isdir(dir_path+"model"):
    model = tf.keras.models.load_model(dir_path+"model")
    print("successfully loaded model from " + dir_path)

#redo training
else:
    mnist = tf.keras.datasets.mnist
    
    (train_set, train_labels), (test_set, test_labels) = mnist.load_data()
    
    #normalise to [0,1]
    train_set, test_set = train_set/255, test_set/255
    
    if diagnostic:
        print(train_set.shape)
        
        
    if diagnostic:
        plt.figure()
        plt.imshow(train_set[0])
        plt.xlabel(train_labels[0])
        plt.show()

    

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    model.fit(train_set, train_labels, epochs=5)
    
    test_loss, test_acc = model.evaluate(test_set, test_labels, verbose=2)
    
    print("Accuracy:",test_acc)

    #done training
    
    model.save("model")
    print("successfully saved model to " + dir_path)

    
prob_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

if diagnostic:
    print(test_set[0])




    
#shell mode
while True:

    filename = input("\nInput .bmp file name OR test_set index (numeric)? (or exit)")
    if filename == "exit":
        break

    try:
        index = int(filename)
        mode = "test_set"
    except ValueError:
        mode = "bmp"


    if mode == "bmp":
    
        if len(filename) <= 4 or filename[-4:] != ".bmp":
            filename = dir_path + filename + ".bmp"

        print("Loading",filename)
        try:
            img = Image.open(filename)
        except IOError:
            print(filename,"could not be loaded or does not exist")
            continue

        raw_array = np.asarray(img)
        
        if diagnostic:
            print(raw_array.shape)

        if raw_array.shape[:2] != (28,28):
            raise ValueError("bmp size must be 28*28")
          
        #currently rawArray is a 28*28*3 array (rgb), want only one channel, normalize, and invert it so that black is 1, white is 0.
        img_array = np.array(
            [[1-(sum(raw_array[i][j])//len(raw_array[0][0]))/255 for j in range(len(raw_array[0]))] for i in range(len(raw_array))]
        )

    elif mode == "test_set":
        try:
            img_array = test_set[index]
        except NameError:
            #test_set was not defined because model loaded from dir
            #recall mnist
            mnist = tf.keras.datasets.mnist
    
            (train_set, train_labels), (test_set, test_labels) = mnist.load_data()
            
            #normalise to [0,1]
            train_set, test_set = train_set/255, test_set/255

            img_array = test_set[index]
    
            
    plt.figure()
    plt.imshow(img_array)
    plt.show()
    
    img = (np.expand_dims(img_array,0))

    predictions = prob_model.predict(img)
    
    print("Predictions:")
    print(predictions)
    print("Most likely a:",np.argmax(predictions))
