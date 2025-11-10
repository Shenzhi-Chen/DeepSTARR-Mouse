# Following links were used to prepare this script. 
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# https://github.com/igul222/improved_wgan_training
# https://arxiv.org/abs/1712.06148

from __future__ import print_function, division
import os
import errno
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, add, Activation
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from functools import partial
import keras.backend as K
import numpy as np


BATCH_SIZE = 128
ITERS = 400001
SEQ_LEN = 500
SEQ_DIM = 4
DIM = 128
CRITIC_ITERS = 10
LAMBDA = 1
loginterval = 1000
seqinterval = 10000
modelinterval = 10000
selectedmodel = 400000
suffix = "generated"
ngenerate = 10
outputdirc = "./output/"
fastafile = "./data/KC_regions.fa"


for file in [outputdirc,
             os.path.join(outputdirc, 'models'),
             os.path.join(outputdirc, 'samples_ACGT'),
             os.path.join(outputdirc, 'samples_raw')]:
    try:
        os.makedirs(file)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass


def readfile(filename):
    ids = []
    seqs = []
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    seq = []
    for line in lines:
        if line[0] == '>':
            ids.append(line[1:].rstrip('\n'))
            if seq != []: seqs.append("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq != []:
        seqs.append("".join(seq))

    return ids, seqs


def one_hot_encode_along_row_axis(sequence):
    to_return = np.zeros((1, len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return[0],
                                 sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    for (i, char) in enumerate(sequence):
        if char == "A" or char == "a":
            char_idx = 0
        elif char == "C" or char == "c":
            char_idx = 1
        elif char == "G" or char == "g":
            char_idx = 2
        elif char == "T" or char == "t":
            char_idx = 3
        elif char == "N" or char == "n":
            continue
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((BATCH_SIZE, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():
    def __init__(self):
        self.img_rows = SEQ_LEN
        self.img_cols = SEQ_DIM
        self.img_shape = (self.img_rows, self.img_cols)
        self.latent_dim = DIM

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = CRITIC_ITERS
        optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(DIM,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(DIM,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = LAMBDA * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def res_cnn(self):
        input_tensor = Input(shape=(SEQ_LEN, DIM))
        x = Activation('relu')(input_tensor)
        x = Conv1D(DIM, 5, padding='same')(x)
        output = add([input_tensor, x])
        res_1d = Model(inputs=[input_tensor], outputs=[output])
        return res_1d

    def build_generator(self):
        model = Sequential()
        model.add(Dense(SEQ_LEN * DIM, activation='elu', input_shape=(DIM,)))
        model.add(Reshape((SEQ_LEN, DIM)))
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(Conv1D(SEQ_DIM, 1, padding='same'))
        model.add(Activation('softmax'))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_critic(self):
        model = Sequential()
        model.add(Conv1D(DIM, 1, padding='same', input_shape=(SEQ_LEN, SEQ_DIM)))
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(self.res_cnn())
        model.add(Flatten())
        model.add(Dense(1))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, foldername, filename, epochs, batch_size,
              log_interval=1000, seq_interval=10000, model_interval=10000):

        ids, seqs = readfile(filename)
        X_train = np.array([one_hot_encode_along_row_axis(seq) for seq in seqs]).squeeze(axis=1)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))

        disc_json = self.critic_model.to_json()
        with open(foldername + '/disc.json', "w") as disc_json_file:
            disc_json_file.write(disc_json)

        gen_json = self.generator_model.to_json()
        with open(foldername + '/gen.json', "w") as gen_json_file:
            gen_json_file.write(gen_json)
            
        d_loss_list = []
        g_loss_list = []
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                          [valid, fake, dummy])
            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.generator_model.train_on_batch(noise, valid)

            if epoch % log_interval == 0:
                d_loss_list.append(d_loss)
                g_loss_list.append(g_loss)

            if epoch % seq_interval == 0:
                samples = []
                for i in range(1):
                    samples.extend(self.generate_samples())
                with open(foldername + '/samples_ACGT/samples_ACGT_{}.fa'.format(epoch), 'w') as f:
                    for line_number, s in enumerate(samples[0]):
                        f.write(">" + str(line_number+1) + "\n")
                        s = "".join(s)
                        f.write(s + "\n")
                with open((foldername + '/samples_raw/samples_{}.txt').format(epoch), 'w') as f2:
                    print(samples[1], file=f2)

            if epoch % model_interval == 0:
                self.critic_model.save_weights(foldername + '/models/disc_{}.hdf5'.format(epoch))
                self.critic_model.save(foldername + '/models/disc_{}.h5'.format(epoch))
                self.generator_model.save_weights(foldername + '/models/gen_{}.hdf5'.format(epoch))
                self.generator_model.save(foldername + '/models/gen_{}.h5'.format(epoch))
        
        
        import pickle
        f = open(foldername + '/d_g_loss.pkl', "wb")
        pickle.dump(d_loss_list,f)
        pickle.dump(g_loss_list,f)
        f.close()
        

    def generate_samples(self):
        char_ACGT={0:'A' , 1:'C' , 2:'G' , 3:'T'}
        noise = np.random.normal(0, 1, (BATCH_SIZE, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        samples = np.argmax(gen_imgs, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = ''
            for j in range(len(samples[i])):
                decoded += char_ACGT[samples[i][j]]
            decoded_samples.append(decoded)
        return decoded_samples, gen_imgs

    def generate(self, nb=1, model_number=0, result_number=0):
        hdf5_filename = outputdirc + "/models/disc_" + str(model_number) + ".hdf5"
        self.generator_model.load_weights(hdf5_filename)
        samples = []
        for i in range(nb):
            samples.extend(self.generate_samples()[0])
        with open(outputdirc + '/gen_seq/generated_{}_iter_{}.fa'.format(nb*BATCH_SIZE, model_number), 'w') as f:
            counter = 0
            for s in samples:
                counter += 1
                s = "".join(s)
                f.write(">" + str(counter) + "_" + str(result_number) + "_" + str(model_number) + "\n" + s + "\n")


if __name__ == '__main__':
    wgan = WGANGP()
    # Train the model
    wgan.train(outputdirc, fastafile, epochs=ITERS, batch_size=BATCH_SIZE,
               log_interval=loginterval, seq_interval=seqinterval, model_interval=modelinterval)
    
    # Generate sequences after training
    try:
        os.makedirs(os.path.join(outputdirc, 'gen_seq'))
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
    for i in range(0, selectedmodel+1, modelinterval):
        wgan.generate(nb=ngenerate, model_number=i, result_number=suffix)
