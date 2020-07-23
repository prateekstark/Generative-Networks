from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.autograd import Variable
from models import *
from utils import *
import argparse
import logging

logging.basicConfig(filename='logfile.log', format='%(levelname)s %(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help="Default = 32", type=int, default=32)
parser.add_argument('--epochs', help="Default = 100", type=int, default=100)
parser.add_argument('--lr', help="Default = 0.001", type=float, default=0.001)
parser.add_argument('--save_loss_counter', help="Default = True", type=bool, default=True)
parser.add_argument('--latent_space_size', help="Default = 8", type=int, default=8)

if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
    Tensor = torch.cuda.FloatTensor
    logger.info("device: cuda:0")
else:
    device = torch.device("cpu")
    Tensor = torch.FloatTensor
    logger.info("device: cpu")

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
SAVE_LOSS_COUNTER = args.save_loss_counter
LEARNING_RATE = args.lr
LATENT_SPACE_SIZE = args.latent_space_size

logger.info("EPOCHS: " + str(EPOCHS))
logger.info("BATCH_SIZE: " + str(BATCH_SIZE))
logger.info("SAVE_LOSS_COUNTER: " + str(SAVE_LOSS_COUNTER))
logger.info("LEARNING_RATE: " + str(LEARNING_RATE))
logger.info("LATENT_SPACE_SIZE: " + str(LATENT_SPACE_SIZE))

train = datasets.MNIST('', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()       
                      ]))

trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

logger.info("Data loaded...")
    
def eval(decoder, epoch, z_dim):
    decoder.eval()
    with torch.no_grad():
        sample = torch.randn(64, z_dim).to(device)
        sample = decoder(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), 'image_' + str(epoch) + '.png')

encoder = Encoder(LATENT_SPACE_SIZE).to(device)
decoder = Decoder(LATENT_SPACE_SIZE).to(device)
discriminator = Discriminator(LATENT_SPACE_SIZE).to(device)

logger.info("Loaded Encoder, Decoder and Discriminator...")

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

logger.info("encoder_optim: " + str(encoder_optimizer))
logger.info("decoder_optim: " + str(decoder_optimizer))
logger.info("discriminator_optim: " + str(discriminator_optimizer))

discriminator_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

logger.info("discriminator_loss: " + str(discriminator_loss))
logger.info("reconstruction_loss: " + str(reconstruction_loss))

D_loss_counter = []
AE_loss_counter = []

encoder.train()
discriminator.train()
for epoch in range(EPOCHS):
    logger.info("epoch: " + str(epoch))
    for data in trainset:
        X, _ = data
        X = X.view(-1, 784).to(device)
        
        true = Variable(Tensor(BATCH_SIZE, 1).fill_(1.0))
        fake = Variable(Tensor(BATCH_SIZE, 1).fill_(0.0))
        
        noise = torch.randn((BATCH_SIZE, LATENT_SPACE_SIZE), device=device)
        gen_z = encoder(X)

        X_all = torch.cat((noise, gen_z), dim=0)
        y_all = torch.cat((true, fake), dim=0)

        discriminator_optimizer.zero_grad()
        output = discriminator(X_all)
        disc_loss = discriminator_loss(output, y_all)
        disc_loss.backward()
        discriminator_optimizer.step()
        D_loss_counter.append(disc_loss)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        latent_z = encoder(X)
        reconstructed_imgs = decoder(latent_z)
        autoencoder_loss = 0.999 * reconstruction_loss(reconstructed_imgs, X) + 0.001 * discriminator_loss(discriminator(latent_z), true)
        autoencoder_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        AE_loss_counter.append(autoencoder_loss)

    if(epoch%2 == 0):
        eval(decoder, epoch, LATENT_SPACE_SIZE)

if(SAVE_LOSS_COUNTER):
    torch.save(D_loss_counter, "G_loss_counter.pt")
    torch.save(AE_loss_counter, "AE_loss_counter.pt")
    plot_loss_counter(AE_loss_counter, D_loss_counter)
    logger.info("loss-counter curve saved...")

make_gif()
logger.info("Eval GIF saved...")