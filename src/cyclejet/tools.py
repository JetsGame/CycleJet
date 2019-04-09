import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

def plot_model(fn, refA, refB, predictedA, predictedB):
    """Plot the results for reference and predictions"""
    with PdfPages(fn) as pdf:
        fig, axs = plt.subplots(2,2, figsize=(6,7))
        axs[0,0].imshow(np.average(refA,axis=0)[:,:,0],vmin=0.0,vmax=0.7)
        axs[0,0].set_title('average A sample')
        axs[0,1].imshow(np.average(predictedA,axis=0)[:,:,0],vmin=0.0,vmax=0.7)
        axs[0,1].set_title('average reconstructed')
        axs[1,0].imshow(np.average(refB,axis=0)[:,:,0],vmin=0.0,vmax=0.7)
        axs[1,0].set_title('average B sample')
        axs[1,1].imshow(np.average(predictedB,axis=0)[:,:,0],vmin=0.0,vmax=0.7)
        axs[1,1].set_title('average reconstructed')
        plt.close()
        pdf.savefig(fig)
