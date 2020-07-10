from torchvision import transforms, datasets
from torchvision.utils import save_image
from models import *
from utils import *
import argparse
import logging


logging.basicConfig(filename='logfile.log', format='%(levelname)s %(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help="Default = 100", type=int, default=100)
parser.add_argument('--epochs', help="Default = 10", type=int, default=10)
parser.add_argument('--lr', help="Default = 0.001", type=float, default=0.001)
parser.add_argument('--save_loss_counter', help="Default = True", type=bool, default=True)
parser.add_argument('--latent_space_size', help="Default = 20", type=int, default=20)

if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
    logger.info("device: cuda:0")
else:
    device = torch.device("cpu")
    logger.info("device: cpu")

args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
SAVE_LOSS_COUNTER = args.save_loss_counter
LEARNING_RATE = args.lr
LATENT_SPACE_SIZE = args.latent_space_size


train = datasets.MNIST('', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()       
                      ]))


trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

logging.info("Data loaded...")
logging.info("EPOCHS: " + str(EPOCHS))
logging.info("BATCH_SIZE: " + str(BATCH_SIZE))
logging.info("SAVE_LOSS_COUNTER: " + str(SAVE_LOSS_COUNTER))
      
def vae_loss(output, target, mu, logvariance):
    BCEloss = F.binary_cross_entropy(output, target, reduction='sum')
    KL_Divergence = -0.5 * torch.sum(1 + logvariance - mu.pow(2) -logvariance.exp())
    return BCEloss + KL_Divergence

def eval(vae, epoch, z_dim):
    vae.eval()
    with torch.no_grad():
        sample = torch.randn(64, z_dim).to(device)
        sample = vae.decoder(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), 'image_' + str(epoch) + '.png')
    
vae = VAE(LATENT_SPACE_SIZE).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
logging.info("optimizer: Adam")
logging.info("LEARNING_RATE: " + str(LEARNING_RATE))

loss_counter = []
total_loss = 0
counter = 1

vae.train()
for epoch in range(EPOCHS):
    logging.info("epoch: " + str(epoch))
    for data in trainset:
        X, _ = data
        X = X.view(-1, 784).to(device)
        optimizer.zero_grad()
        output, mu, logvariance = vae(X)
        loss = vae_loss(output, X, mu, logvariance)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        moving_average = total_loss/counter
        if(counter%100 == 0):
            logging.info(str(counter) + ": " + str(moving_average))
        loss_counter.append(moving_average)
        counter += 1
    eval(vae, epoch, LATENT_SPACE_SIZE)

if(SAVE_LOSS_COUNTER):
    import pickle
    with open('loss.pkl', 'wb') as f:
        pickle.dump(loss_counter, f)
    plot_loss_counter(loss_counter)
    logging.info("loss counter curve saved...")

make_gif()
logging.info("Eval GIF saved...")