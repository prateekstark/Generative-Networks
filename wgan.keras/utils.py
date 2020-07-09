import matplotlib.pyplot as plt
import imageio
import glob
import math
from IPython import display

## Source https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif
def generate_and_save_images(model, epoch, test_input, save=True):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    maxLen = int(math.sqrt(test_input.shape[0]))
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))

    for i in range(maxLen*maxLen):
        plt.subplot(maxLen, maxLen, i+1)
        plt.imshow((predictions[i, :, :, :] + 1.0) / 2.0)
        plt.axis('off')
    if(save):
        plt.savefig('image_at_epoch_{:07d}.png'.format(epoch))
    # plt.show()


def make_gif():
    anim_file = 'wgan.gif'

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
