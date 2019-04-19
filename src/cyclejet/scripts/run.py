import argparse, yaml, pprint, os, shutil, datetime, sys, pickle
import numpy as np
from time import time
from cyclejet.cyclegan import CycleGAN
from cyclejet.tools import loss_calc, plot_model
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from hyperopt.mongoexp import MongoTrials
import keras.backend as K


#----------------------------------------------------------------------
def run_hyperparameter_scan(search_space, max_evals, cluster, folder):
    """Running hyperparameter scan using hyperopt"""
    print('[+] Performing hyperparameter scan...')
    if cluster:
        trials = MongoTrials(cluster, exp_key='exp1')
    else:
        trials = Trials()
    best = fmin(build_and_train_model, search_space, algo=tpe.suggest, 
                max_evals=max_evals, trials=trials)
    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)
    log = '%s/hyperopt_log_{}.pickle'.format(time()) % folder
    with open(log, 'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)
    return best_setup


#----------------------------------------------------------------------
def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, 'r') as stream:
        runcard = yaml.load(stream)
    for key, value in runcard.items():
        if 'hp.' in str(value):
            runcard[key] = eval(value)
    return runcard


#----------------------------------------------------------------------
def build_and_train_model(hps):
    """Training model"""
    print('[+] Training model')
    K.clear_session()
    cgan = CycleGAN(hps)

    print('[+] Training CycleGAN')
    cgan.train(epochs=hps['epochs'], batch_size=hps['batch_size'])

    print('[+] Creating test set and saving to file') 

    refA=np.array(cgan.imagesA)
    refB=np.array(cgan.imagesB)
    # generating predicted sample
    predictA=cgan.g_BA.predict(refA)
    predictB=cgan.g_AB.predict(refB)
    # now calculate the loss
    loss = loss_calc(refA, refB, predictA, predictB)
    # now inverse the preprocessing step
    refA = cgan.preproc.inverse(refA)
    refB = cgan.preproc.inverse(refB)
    predictA = cgan.preproc.inverse(predictA)
    predictB = cgan.preproc.inverse(predictB)
    # refA = cgan.avg.inverse(refA)
    # refB = cgan.avg.inverse(refB)
    # predictA = cgan.avg.inverse(predictA)
    # predictB = cgan.avg.inverse(predictB)
    # save the model weights
    if hps['scan']:
        res = {'loss': loss, 'status': STATUS_OK}
    else:
        res = cgan, refA, refB, predictA, predictB, loss
    return res


def main():
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train a cycleGAN.')
    parser.add_argument('runcard', action='store', default=None,
                        help='A json file with the setup.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='The output folder.', required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--savefull', action='store_true')
    parser.add_argument('--hyperopt', default=None, type=int,
                        help='Enable hyperopt scan.')
    parser.add_argument('--cluster', default=None, type=str, 
                        help='Enable cluster scan.')
    args = parser.parse_args()

    # check input is coherent
    if not os.path.isfile(args.runcard):
        raise ValueError('Invalid runcard: not a file.')
    if args.force:
        print('WARNING: Running with --force option will overwrite existing model.')

    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')
    out = args.output.strip('/')

    # copy runcard to output folder
    shutil.copyfile(args.runcard, f'{out}/input-runcard.json')

    print('[+] Loading runcard')
    hps=load_yaml(args.runcard)

    if args.hyperopt:
        hps['scan'] = True
        hps = run_hyperparameter_scan(hps, args.hyperopt, args.cluster, out)
    hps['scan'] = False

    cgan, refA, refB, predictA, predictB, loss = build_and_train_model(hps)

    cgan.save(out)
    if args.savefull:
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

    # now create diagnostic plots
    figfn='%s/result.pdf' % out
    plot_model(figfn, refA, refB, predictA, predictB)
