import matplotlib.pyplot as plt

def plot_history(hist):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='val')
    plt.legend(); plt.title('Accuracy'); plt.show()
