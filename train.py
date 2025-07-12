from data_loader import get_data_generators
from model import build_model
from utils import plot_history
import os
import tensorflow as tf

def main():
    train_dir = os.path.join('data','train')
    val_dir   = os.path.join('data','val')
    train_ds, val_ds = get_data_generators(train_dir, val_dir)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    hist = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
    plot_history(hist)

if __name__ == '__main__':
    main()
