import argparse, json, pprint, os, shutil, datetime, sys
import numpy as np
from cyclegan_jets import CycleGAN
from tools import loss_calc

# read command line arguments
parser = argparse.ArgumentParser(description='Train a cycleGAN.')
parser.add_argument('runcard', action='store', default=None,
                    help='A json file with the setup.')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='The output folder.')
parser.add_argument('--force', action='store_true')
parser.add_argument('--light', action='store_true')

args = parser.parse_args()

# check that options are valid
base = os.path.basename(args.runcard)
out = args.output if args.output else os.path.splitext(base)[0]
if os.path.exists(out) and not args.force:
    raise Exception('Output folder %s already exists.' % out)

print('[+] Loading runcard')
with open(args.runcard,'r') as f:
    hps=json.load(f)
pprint.pprint(hps)
cgan = CycleGAN(hps)

print('[+] Training CycleGAN')
cgan.train(epochs=hps['epochs'], batch_size=hps['batch_size'])

print('[+] Creating test set and saving to file') 

refA=cgan.sampleA
refB=cgan.sampleB
# generating predicted sample
predictA=cgan.g_BA.predict(refA)
predictB=cgan.g_AB.predict(refB)
# now calculate the loss
loss = loss_calc(refA, refB, predictA, predictB)

# set up the output folder
if not os.path.exists(out):
    os.mkdir(out)
elif args.force:
    print(f'WARNING: Overwriting {out} with new model')
    shutil.rmtree(out)
    os.mkdir(out)
else:
    raise Exception('Output folder %s already exists.' % out)

# copy runcard to output folder
shutil.copyfile(args.runcard, f'{out}/input-runcard.json')

# save the model weights
cgan.save(out)
if not args.light:
    np.save('%s/referenceA'%out, refA)
    np.save('%s/referenceB'%out, refB)
    np.save('%s/predictedA'%out, predictA)
    np.save('%s/predictedB'%out, predictB)
# write out a file with basic information on the run
with open('%s/info.txt' % out,'w') as f:
    print('# CyleGAN %s to %s' % (hps['labelA'],hps['labelB']), file=f)
    print('# created on %s with the command:'
          % datetime.datetime.utcnow(), file=f)
    print('# '+' '.join(sys.argv), file=f)
    print('# loss = %f' % loss, file=f)
