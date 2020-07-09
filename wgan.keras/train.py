from WGAN import *
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--steps', help="Default = 50K", type=int, default=50000)
parser.add_argument('--batch_size', help="Default = 32", type=int, default=32)
parser.add_argument('--num_features', help="Default = 100", type=int, default=100)
parser.add_argument('--large_dataset', help='Default = False', type=bool, default=False)
parser.add_argument('--load_model', help='Default = None', default=None)

args = parser.parse_args()

steps = args.steps
BATCH_SIZE = args.batch_size
NUM_FEATURES = args.num_features
LARGE_DATASET = args.large_dataset
DATA_DIRECTORY = 'images'
model = args.load_model

x_sampler = DataSampler(DATA_DIRECTORY, LARGE_DATASET)
z_sampler = NoiseSampler(NUM_FEATURES)
SEED = z_sampler.sample(BATCH_SIZE)

logging.basicConfig(filename="logfile.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info("WGAN initialized...")
logger.info("BATCH_SIZE: " + str(BATCH_SIZE))
logger.info("NUM_FEATURES: " + str(NUM_FEATURES))
logger.info("steps: " + str(steps))
logger.info("LARGE_DATASET: " + str(LARGE_DATASET))

WGAN = WassersteinGAN(x_sampler, z_sampler, NUM_FEATURES, logger)
if(model):
	WGAN.load_model(model)

WGAN.train(BATCH_SIZE, SEED, steps, 5)
make_gif()