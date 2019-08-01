#!/usr/bin/env python
# This file is part of CycleJet by S. Carrazza and F. A. Dreyer

import argparse, yaml, pprint, os, shutil, datetime, sys, pickle
import numpy as np
from cyclejet.cyclegan import CycleGAN
from cyclejet.tools import loss_calc, plot_model
from cyclejet.scripts.run import load_yaml

def main(args):
    if os.path.isfile(args.model.strip('/')+'/best-model.yaml'):
        fn=args.model.strip('/')+'/best-model.yaml'
    else:
        fn=args.model.strip('/')+'/input-runcard.json'
    hps=load_yaml(fn)
    cgan = CycleGAN(hps)
    cgan.load(args.model.strip('/'))
    refA=np.array(cgan.imagesA)
    refB=np.array(cgan.imagesB)
    # generating predicted sample
    predictA=cgan.g_BA.predict(refA)
    predictB=cgan.g_AB.predict(refB)
    refA = cgan.preproc.inverse(refA)
    refB = cgan.preproc.inverse(refB)
    predictA = cgan.preproc.inverse(predictA)
    predictB = cgan.preproc.inverse(predictB)
    if args.savefull:
        np.save('%s/referenceA'%args.model.strip('/'), refA)
        np.save('%s/referenceB'%args.model.strip('/'), refB)
        np.save('%s/predictedA'%args.model.strip('/'), predictA)
        np.save('%s/predictedB'%args.model.strip('/'), predictB)
    
    # now create diagnostic plots
    figfn='%s/result.pdf' % args.model.strip('/')
    plot_model(figfn, refA, refB, predictA, predictB,
               titleA=args.titleA, titleB=args.titleB)

#----------------------------------------------------------------------
if __name__ == "__main__":
    """read command line arguments"""
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train a cycleGAN.')
    parser.add_argument('model', action='store', default=None,
                        help='A folder with the model.')
    parser.add_argument('--titleA', type=str, default=None,
                        help='Title of sample A.')
    parser.add_argument('--titleB', type=str, default=None,
                        help='Title of sample A.')

    parser.add_argument('--savefull', action='store_true')
    args = parser.parse_args()
    main(args)
