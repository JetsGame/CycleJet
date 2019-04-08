import argparse, json
from cyclegan_jets import CycleGAN

# read command line arguments
parser = argparse.ArgumentParser(description='Train a cycleGAN.')
parser.add_argument('runcard', action='store', default=None,
                    help='A json file with the setup.')
args = parser.parse_args()

with open(args.runcard,'r') as f:
    hps=json.load(f)
print(hps)
gan = CycleGAN(hps)
gan.train(epochs=hps['epochs'], batch_size=hps['batch_size'],
          sample_interval=2000)
