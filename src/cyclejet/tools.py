# This file is part of CycleJet by S. Carrazza and F. A. Dreyer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

xval = [0.0, 7.0]
yval = [-3.0, 7.0]

#----------------------------------------------------------------------
def loss_calc(refA, refB, predictedA_fromB, predictedB_fromA):
    """

    The loss is defined by taking the difference of the average image before and 
    after transformation. For two samples A and B, it is then:
    L = norm(<reference A> - <predicted A from B>) 
        + norm(<reference B> - <predicted B from A>)
    """
    img_lossA = np.linalg.norm(np.average(refA,axis=0)-np.average(predictedA_fromB,axis=0))
    img_lossB = np.linalg.norm(np.average(refB,axis=0)-np.average(predictedB_fromA,axis=0))
    loss = img_lossA + img_lossB
    print('Loss = %.4f  (A: %.4f, B: %.4f)' % (loss, img_lossA, img_lossB))
    return loss

#----------------------------------------------------------------------
def plot_model(fn, refA, refB, predictedA, predictedB, titleA=None, titleB=None):
    """Plot the results for reference and predictions"""
    with PdfPages(fn) as pdf:
        
        fig, axs = plt.subplots(2,2, figsize=(6,7))
        plt.subplots_adjust(wspace=0.33,hspace=0.35)
        axs[0,0].imshow(np.average(refA,axis=0).transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        axs[0,0].set_xlabel('$\ln(1 / \Delta_{ab})$')
        axs[0,0].set_ylabel('$\ln(k_{t} / \mathrm{GeV})$',labelpad=-2)
        axs[0,0].set_title('average A sample' if not titleA else titleA)
        axs[0,1].imshow(np.average(predictedB,axis=0).transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        axs[0,1].set_xlabel('$\ln(1 / \Delta_{ab})$')
        axs[0,1].set_ylabel('$\ln(k_{t} / \mathrm{GeV})$',labelpad=-2)
        axs[0,1].set_title('transformed')
        axs[1,0].imshow(np.average(refB,axis=0).transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        axs[1,0].set_xlabel('$\ln(1 / \Delta_{ab})$')
        axs[1,0].set_ylabel('$\ln(k_{t} / \mathrm{GeV})$',labelpad=-2)
        axs[1,0].set_title('average B sample' if not titleB else titleB)
        axs[1,1].imshow(np.average(predictedA,axis=0).transpose(),vmin=0.0,vmax=0.2,origin='lower',
                        aspect='auto', extent=[xval[0], xval[1], yval[0], yval[1]])
        axs[1,1].set_xlabel('$\ln(1 / \Delta_{ab})$')
        axs[1,1].set_ylabel('$\ln(k_{t} / \mathrm{GeV})$',labelpad=-2)
        axs[1,1].set_title('transformed')
        pdf.savefig()
        plt.close()
        
        fig, axs = plt.subplots(3, 2, figsize=(6,8))

        axs[0,0].set_title('A sample')
        axs[0,0].imshow(refA[0])
        axs[1,0].imshow(refA[1])
        axs[2,0].imshow(refA[2])

        axs[0,1].set_title('Transformed')
        axs[0,1].imshow(predictedB[0])
        axs[1,1].imshow(predictedB[1])
        axs[2,1].imshow(predictedB[2])
        pdf.savefig()
        plt.close()
        
        fig, axs = plt.subplots(3, 2, figsize=(6,8))

        axs[0,0].set_title('B sample')
        axs[0,0].imshow(refB[0])
        axs[1,0].imshow(refB[1])
        axs[2,0].imshow(refB[2])

        axs[0,1].set_title('Transformed')
        axs[0,1].imshow(predictedA[0])
        axs[1,1].imshow(predictedA[1])
        axs[2,1].imshow(predictedA[2])
        plt.close()
        pdf.savefig(fig)
