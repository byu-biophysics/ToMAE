import matplotlib.pyplot as plt
import os

# Finish writing plot loss script
def plot_loss(train_loss: list, fpath: str, test_loss: list=[], **kwargs):
    plt.plot(train_loss, label="Training Loss")

    if test_loss:
        plt.plot(test_loss, label="Testing Loss")
    
    plt.title("Loss Per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(fpath,"loss_graph.png"))