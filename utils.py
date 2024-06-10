# plot loss-acc-f1score curves
def plot_loss_curves(results, figsize=(12, 3)):
    # Define the figure size (width, height)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # loss curves
    epochs = range(len(results['train_loss']))
    axes[0].plot(epochs, results['train_loss'], label='train_loss')
    axes[0].plot(epochs, results['val_loss'], label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='best')

    # acc curves
    axes[1].plot(epochs, results['train_acc'], label='train_acc')
    axes[1].plot(epochs, results['val_acc'], label='val_acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(loc='best')