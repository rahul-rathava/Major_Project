import tensorflow as tf

def get_data_generators(train_dir, val_dir, img_size=(224,224), batch=32):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20, width_shift_range=0.1, horizontal_flip=True
    )
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch, class_mode='binary')
    val_ds   = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch, class_mode='binary')
    return train_ds, val_ds
