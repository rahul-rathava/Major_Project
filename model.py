import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionResNetV2

def build_model(img_size=(224,224,3), lr=1e-4):
    base = InceptionResNetV2(include_top=False, input_shape=img_size, weights='imagenet')
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model
