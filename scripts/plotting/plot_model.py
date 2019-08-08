#!/usr/bin/env python
# This file is part of CycleJet by S. Carrazza and F. A. Dreyer

import argparse, yaml, pprint, os, shutil, datetime, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
from cyclejet.cyclegan import CycleGAN
from cyclejet.tools import loss_calc, plot_model, xval, yval
from cyclejet.scripts.run import load_yaml
from random import randrange

def plot_event(fn, refA, refB, predictA, predictB,
               predictA2, predictB2, averager, titleA=None, titleB=None):
    with PdfPages(fn) as pdf:
        fig, axs = plt.subplots(2,3, figsize=(7.5,5.5))
        plt.subplots_adjust(wspace=0.5,hspace=0.52)
        i = randrange(len(refA))
        # figtr = fig.transFigure.inverted()
        # ptB = figtr.transform(ax0tr.transform((225., -10.)))
        # ptE = figtr.transform(ax1tr.transform((225., 1.)))
        # arrow=FancyArrowPatch(
        #     ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
        #     fc = "g", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha = 0.3,
        #     mutation_scale = 40.)
        # fig.patches.append(arrow)
        axs[0,0].imshow(refA[i].transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        axs[0,0].set_title('A' if not titleA else titleA)
        axs[0,0].set_xticks([])
        axs[0,0].set_yticks([])
        axs[0,1].imshow(predictB[i].transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        # axs[0,1].set_xlabel('$\ln(1 / \Delta_{ab})$')
        # axs[0,1].set_ylabel('$\ln(k_{t} / \mathrm{GeV})$',labelpad=-2)
        axs[0,1].set_title('B' if not titleB else titleB)
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,2].imshow(predictB2[i].transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        axs[0,2].set_title('A' if not titleA else titleA)
            
        axs[0,2].set_xticks([])
        axs[0,2].set_yticks([])
        axs[1,0].imshow(averager.inverse(refA)[i].transpose(),
                        vmin=0.0,vmax=0.2,origin='lower',aspect='auto',
                        extent=[xval[0], xval[1], yval[0], yval[1]])
        #axs[1,0].set_title('A' if not titleA else titleA)
        axs[1,0].set_xticks([])
        axs[1,0].set_yticks([])
        axs[1,1].imshow(averager.inverse(predictB)[i].transpose(),
                        vmin=0.0,vmax=0.2,origin='lower',aspect='auto',
                        extent=[xval[0], xval[1], yval[0], yval[1]])
        # axs[0,1].set_xlabel('$\ln(1 / \Delta_{ab})$')
        # axs[0,1].set_ylabel('$\ln(k_{t} / \mathrm{GeV})$',labelpad=-2)
        #axs[1,1].set_title('B' if not titleB else titleB)
        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([])
        axs[1,2].imshow(averager.inverse(predictB2)[i].transpose(),
                        vmin=0.0,vmax=0.2,origin='lower',aspect='auto',
                        extent=[xval[0], xval[1], yval[0], yval[1]])
        #axs[1,2].set_title('A' if not titleA else titleA)
        axs[1,2].set_xticks([])
        axs[1,2].set_yticks([])
        # pdf.savefig()
        # plt.close()
        
        # fig, axs = plt.subplots(3, 2, figsize=(6,8))
        plt.close()
        pdf.savefig(fig)
    

def main(args):
    model=args.model.strip('/')
    if os.path.isfile(model+'/best-model.yaml'):
        fn=model+'/best-model.yaml'
    else:
        fn=model+'/input-runcard.json'
    hps=load_yaml(fn)
    cgan = CycleGAN(hps)
    cgan.load(model)
    refA=np.array(cgan.imagesA)
    refB=np.array(cgan.imagesB)
    # generating predicted sample
    predictA=cgan.g_BA.predict(refA)
    predictB=cgan.g_AB.predict(refB)
    predictA2=cgan.g_AB.predict(predictA)
    predictB2=cgan.g_BA.predict(predictB)
    refA = cgan.preproc.inverse(refA)
    refB = cgan.preproc.inverse(refB)
    predictA = cgan.preproc.inverse(predictA)
    predictB = cgan.preproc.inverse(predictB)
    if args.savefull:
        np.save('%s/referenceA'%model, refA)
        np.save('%s/referenceB'%model, refB)
        np.save('%s/predictedA'%model, predictA)
        np.save('%s/predictedB'%model, predictB)
    
    # now create plots
    figfn1='%s/result.pdf' % model
    plot_model(figfn1, refA, refB, predictA, predictB,
                   titleA=args.titleA, titleB=args.titleB)
    figfn2='%s/result_event.pdf' % model
    plot_event(figfn2, refA, refB, predictA, predictB,
               predictA, predictB, cgan.avg,
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
