import numpy as np

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
