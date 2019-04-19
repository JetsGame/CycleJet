from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os, datetime, random

from glund.models.optimizer import build_optimizer
from glund.preprocess import PreprocessorZCA, Preprocessor, Averager
from glund.read_data import Jets
from glund.JetTree import JetTree, LundImage

# TODO: add ZCA preprocessor

class CycleGAN():
    def __init__(self, hps):
        # Input shape
        self.img_rows = hps['npixels']
        self.img_cols = hps['npixels']
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # Load the training data
        self.load_data(hps)
        
        # Calculate output shape of D (PatchGAN)
        if (self.img_rows%16==0):
            patch = int(self.img_rows / 2**4)
        else:
            patch = int(self.img_rows / 2**3)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = hps['g_filters']
        self.df = hps['d_filters']

        # Loss weights
        # Cycle-consistency loss
        self.lambda_cycle = hps['lambda_cycle']
        # Identity loss
        self.lambda_id = hps['lambda_id_factor'] * self.lambda_cycle

        #optimizer = Adam(0.0002, 0.5)
        optimizer = build_optimizer(hps)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def save(self, folder):
        """Save CycleGAN weights to file"""
        self.g_AB.save_weights('%s/generatorAB.h5' % folder)
        self.g_BA.save_weights('%s/generatorBA.h5' % folder)

    def load(self, folder):
        """Load CycleGAN from input folder"""
        self.g_AB.load_weights('%s/generatorAB.h5' % folder)
        self.g_BA.load_weights('%s/generatorBA.h5' % folder)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        if (self.img_rows%16==0):
            d4 = conv2d(d3, self.gf*8)

        # Upsampling
        if (self.img_rows%16==0):
            u1 = deconv2d(d4, d3, self.gf*4)
            u2 = deconv2d(u1, d2, self.gf*2)
        else:
            u2 = deconv2d(d3, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        model = Model(d0, output_img)
        model.summary()
        return model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        if (self.img_rows%16==0):
            d4 = d_layer(d3, self.df*8)
        else:
            d4=d3

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        model = Model(img, validity)
        model.summary()
        return model

    def load_data(self, hps):
        self.dataset_name = '%s2%s' % (hps['labelA'], hps['labelB'])
        self.lund = LundImage(npxlx=self.img_rows, npxly=self.img_cols)
        labelA_files = glob('%s/*%s.json.gz' % (hps['data_path'], hps['labelA']))
        labelB_files = glob('%s/*%s.json.gz' % (hps['data_path'], hps['labelB']))
        
        self.imagesA = []
        self.imagesB = []
        for fn in labelA_files:
            reader = Jets(fn, hps['nev'])
            jets = reader.values()
            for j in jets:
                tree = JetTree(j)
                li = self.lund(tree)
                # if np.random.random() > 0.5:
                #     li = np.fliplr(li)
                self.imagesA.append(li[:,:,np.newaxis])
        for fn in labelB_files:
            reader = Jets(fn, hps['nev'])
            jets = reader.values()
            for j in jets:
                tree = JetTree(j)
                li = self.lund(tree)
                # if np.random.random() > 0.5:
                #     li = np.fliplr(li)
                self.imagesB.append(li[:,:,np.newaxis])
        # now do batch averaging
        self.avg = Averager(hps['navg'])
        self.imagesA = self.avg.transform(np.array(self.imagesA))
        self.imagesB = self.avg.transform(np.array(self.imagesB))

        # now create he preprocessors
        if hps['zca']:
            self.preproc = PreprocessorZCA(scaler=True, flatten=False, remove_zero=False)
        else:
            self.preproc = Preprocessor(scaler=True, flatten=False, remove_zero=False)
        # and preprocess the input images
        self.preproc.fit(np.concatenate((self.imagesA,self.imagesB)))
        self.imagesA = self.preproc.transform(self.imagesA)
        self.imagesB = self.preproc.transform(self.imagesB)

    #-------------------------------------------------------------------------------
    def load_batch(self, batch_size):
        self.n_batches = int(min(len(self.imagesA), len(self.imagesB)) / batch_size)
        total_samples = self.n_batches * batch_size
        random.shuffle(self.imagesA)
        random.shuffle(self.imagesB)
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

    #----------------------------------------------------------------------
    def train(self, epochs, batch_size=1, sample_interval=None):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if sample_interval and batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 4, 3

        imgs_A = np.array(random.sample(list(self.imagesA),1))
        imgs_B = np.array(random.sample(list(self.imagesB),1))
        manyimgs_A = self.imagesA
        manyimgs_B = self.imagesB

        # Translate images to the other domain
        fake_B = self.preproc.inverse(self.g_AB.predict(imgs_A))
        fake_A = self.preproc.inverse(self.g_BA.predict(imgs_B))
        manyfake_B = self.preproc.inverse(self.g_AB.predict(manyimgs_A))
        manyfake_A = self.preproc.inverse(self.g_BA.predict(manyimgs_B))
        # Translate back to original domain
        reconstr_A = self.preproc.inverse(self.g_BA.predict(fake_B))
        reconstr_B = self.preproc.inverse(self.g_AB.predict(fake_A))
        manyreconstr_A = self.preproc.inverse(self.g_BA.predict(manyfake_B))
        manyreconstr_B = self.preproc.inverse(self.g_AB.predict(manyfake_A))
        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B,
                                   np.average(manyimgs_A,axis=0).reshape((1,)+manyimgs_A[0].shape),
                                   np.average(manyfake_B,axis=0).reshape((1,)+manyfake_B[0].shape),
                                   np.average(manyreconstr_A,axis=0).reshape((1,)+manyreconstr_A[0].shape),
                                   np.average(manyimgs_B,axis=0).reshape((1,)+manyimgs_B[0].shape),
                                   np.average(manyfake_A,axis=0).reshape((1,)+manyfake_A[0].shape),
                                   np.average(manyreconstr_B,axis=0).reshape((1,)+manyreconstr_B[0].shape)])

        titles = ['Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c, figsize=(10,10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt][:,:,0],vmin=0.0,vmax=0.8)
                if j>0:
                    tit=titles[j-1]
                elif i==0:
                    tit='single A sample'
                elif i==1:
                    tit='single B sample'
                elif i==2:
                    tit='avg A samples'
                elif i==3:
                    tit='avg B samples'
                axs[i, j].set_title(tit)
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


