import matplotlib.pyplot as plt
import imageio
import glob
from IPython import display

def make_gif():
    anim_file = 'gan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i)
            if round(frame) > round(last):
                last = frame
            else:
              continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    display.Image(open(anim_file, 'rb').read())


def plot_loss_counter(G_loss, D_loss):
    plt.plot(G_loss)
    plt.plot(D_loss)
    plt.legend(["Generator Loss", "Discriminator Loss"])
    plt.ylabel('Loss')
    plt.xlabel('Counter')
    plt.savefig('loss_counter.png')