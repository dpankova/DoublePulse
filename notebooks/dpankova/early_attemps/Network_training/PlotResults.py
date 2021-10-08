import matplotlib as mpl
import matplotlib.pyplot as plt

def PlotAccLoss(history,PlotFilename,Legend):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(Legend)
    
    
    ax1.plot(epochs, acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.set_title('Training and validation accuracy')
#    ax1.legend(Legend)

    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.set_title('Training and validation loss')
#    ax2.legend(Legend)

#    plt.figure()

    plt.savefig(PlotFilename)

    #plt.show()
