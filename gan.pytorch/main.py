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
parser.add_argument('--epochs', help="Default = 10", type=int, default=10)
parser.add_argument('--save_loss_counter', help="Default = True", type=bool, default=True)
parser.add_argument('--latent_space_size', help="Default = 100", type=int, default=100)

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
LATENT_SPACE_SIZE = args.latent_space_size

train = datasets.MNIST('', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                      ]))

trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

logger.info("Data loaded...")
logger.info("EPOCHS: " + str(EPOCHS))
logger.info("BATCH_SIZE: " + str(BATCH_SIZE))
logger.info("SAVE_LOSS_COUNTER: " + str(SAVE_LOSS_COUNTER))
logger.info("LATENT_SPACE_SIZE: " + str(LATENT_SPACE_SIZE))

def eval(generator, epoch, z_dim):
    generator.eval()
    with torch.no_grad():
        sample = torch.randn((64, z_dim), device=device)
        sample = generator(sample).cpu() * 0.5 + 0.5
        save_image(sample.view(64, 1, 28, 28), 'image_' + str(epoch) + '.png')

generator = Generator(LATENT_SPACE_SIZE).to(device)
discriminator = Discriminator().to(device)

logger.info("Loaded generator and discriminator...")

G_optim = torch.optim.RMSprop(generator.parameters(), lr=0.001)
D_optim = torch.optim.RMSprop(discriminator.parameters(), lr=0.001)
logger.info("G_optim: " + str(G_optim))
logger.info("D_optim: " + str(D_optim))

criterion = nn.BCELoss()
logger.info("criterion: " + str(criterion))

G_loss = []
D_loss = []

for epoch in range(EPOCHS):
    logger.info("Entering epoch: " + str(epoch+1))
    for data in trainset:
        X, _ = data
        X = (X.view(-1, 784).to(device) * 2.0) - 1.0
        
        true = Variable(Tensor(BATCH_SIZE, 1).fill_(1.0))
        fake = Variable(Tensor(BATCH_SIZE, 1).fill_(0.0))
        noise = torch.randn((BATCH_SIZE, LATENT_SPACE_SIZE), device=device)
        gen_imgs = generator(noise)

        X_all = torch.cat((X, gen_imgs), dim=0)
        y_all = torch.cat((true, fake), dim=0)

        # Training Discriminator
        D_optim.zero_grad()
        loss = criterion(discriminator(X_all), y_all)
        D_loss.append(loss)
        loss.backward()
        D_optim.step()

        # Training Generator
        noise = torch.randn((BATCH_SIZE, LATENT_SPACE_SIZE), device=device)
        G_optim.zero_grad()
        gen_imgs = generator(noise)
        loss = criterion(discriminator(gen_imgs), true)
        G_loss.append(loss)
        loss.backward()
        G_optim.step()
    eval(generator, epoch, LATENT_SPACE_SIZE)

if(SAVE_LOSS_COUNTER):
    torch.save(G_loss, "G_loss.pt")
    torch.save(D_loss, "D_loss.pt")
    plot_loss_counter(G_loss, D_loss)
    logger.info("loss-counter curve saved...")

make_gif()
logger.info("Eval GIF saved...")