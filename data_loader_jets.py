import scipy, random
from itertools import zip_longest
from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, path, labelA, labelB, nev, navg,
                 xval = [0.0, 7.0], yval = [-3.0, 7.0], img_res=(25, 25)):
        self.lund = LundImage(xval, yval, img_res[0], img_res[1], norm_to_one=True)
        self.dataset_path = path
        self.navg = navg
        self.labelA  = labelA
        self.labelB  = labelB
        self.img_res = img_res
        labelA_files = glob('%s/*%s.json.gz' % (path, labelA))
        labelB_files = glob('%s/*%s.json.gz' % (path, labelB))
        print('# reading A from:',labelA_files)
        print('# reading B from:',labelB_files)
        # read in the lund images from files in the path searched
        self.imagesA = []
        self.imagesB = []
        for fn in labelA_files:
            reader = Jets(fn, nev)
            jets = reader.values()
            for j in jets:
                tree = JetTree(j)
                li = self.lund(tree)
                # if np.random.random() > 0.5:
                #     li = np.fliplr(li)
                self.imagesA.append(li[:,:,np.newaxis])
        for fn in labelB_files:
            reader = Jets(fn, nev)
            jets = reader.values()
            for j in jets:
                tree = JetTree(j)
                li = self.lund(tree)
                # if np.random.random() > 0.5:
                #     li = np.fliplr(li)
                self.imagesB.append(li[:,:,np.newaxis])
        # now do batch averaging
        self.imagesA=np.array(self.imagesA)
        self.imagesB=np.array(self.imagesB)
        batch_img_A=[]
        batch_img_B=[]
        for i in range(nev):
            batch_img_A.append(np.average(self.imagesA[np.random.choice(self.imagesA.shape[0], self.navg,
                                                                        replace=False), :], axis=0))
            batch_img_B.append(np.average(self.imagesB[np.random.choice(self.imagesB.shape[0], self.navg,
                                                                        replace=False), :], axis=0))
        self.imagesA = batch_img_A
        self.imagesB = batch_img_B

    def load_data(self, domain, batch_size=1, is_testing=False, nev_test=1000):
        if is_testing:
            files = glob('%s/*%s.json.gz' % (self.dataset_path.replace('train','test'), domain))
            print('# reading test from:',files)
            fn = random.choice(files)
            reader = Jets(fn, nev_test)
            jets = reader.values()
            imgs = []
            for j in jets:
                tree = JetTree(j)
                li = self.lund(tree)
                imgs.append(li[:,:,np.newaxis])
            imgs=np.array(imgs)
            batch_imgs=[]
            for i in range(len(imgs)):
                batch_imgs.append(np.average(imgs[np.random.choice(imgs.shape[0], self.navg,
                                                                   replace=False), :], axis=0))
            imgs=batch_imgs
            imgs=random.sample(imgs, batch_size)
        elif domain==self.labelA:
            imgs = random.sample(self.imagesA, batch_size)
        elif domain==self.labelB:
            imgs = random.choice(self.imagesB, batch_size)
        else:
            raise ValueError ("load_data: Invalid parameter values.")
        
        return np.array(imgs)

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing:
            raise ValueError ("load_batch: validation not implemented")
            # files_A = glob('%s/*%s*' % (self.dataset_path.replace('train','valid'), self.labelA))
            # files_B = glob('%s/*%s*' % (self.dataset_path.replace('train','valid'), self.labelB))
            # fnA = random.choice(files_A)
            # fnB = random.choice(files_B)
            # readerA = Jets(fnA, nev_test)
            # readerB = Jets(fnB, nev_test)
            # jetsA = random.choice(readerA.values(), size=batch_size)
            # jetsB = random.choice(readerB.values(), size=batch_size)

        self.n_batches = int(min(len(self.imagesA), len(self.imagesB)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        random.shuffle(self.imagesA)
        random.shuffle(self.imagesA)

        for i in range(self.n_batches-1):
            batch_A = self.imagesA[i*batch_size:(i+1)*batch_size]
            batch_B = self.imagesB[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                # if not is_testing and np.random.random() > 0.5:
                #         img_A = np.fliplr(img_A)
                #         img_B = np.fliplr(img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            yield np.array(imgs_A), np.array(imgs_B)

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
