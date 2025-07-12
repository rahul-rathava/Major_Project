from tensorflow.keras.models import load_model
from data_loader import get_data_generators
import numpy as np
from sklearn.metrics import classification_report
import os
import tensorflow as tf

def main():
    model = load_model('models/best_model.h5')
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join('data','test'),
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    preds = (model.predict(test_gen)>=0.5).astype(int)
    y_true = test_gen.classes
    print(classification_report(y_true, preds, target_names=list(test_gen.class_indices.keys())))

if __name__ == '__main__':
    main()
