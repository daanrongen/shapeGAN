import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display


class shapeGAN:
    def __init__(
        self,
        batch_size=256,
        gen_lr=1e-4,
        disc_lr=1e-4,
        noise_dim=100,
        checkpoint_dir="./training_checkpoints",
    ):
        self.batch_size = batch_size
        self.noise_dim = noise_dim

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(gen_lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )

        num_examples_to_generate = 16
        self.seed = tf.random.normal([num_examples_to_generate, noise_dim])

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(
            layers.Conv2DTranspose(
                128, (5, 5), strides=(1, 1), padding="same", use_bias=False
            )
        )
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding="same", use_bias=False
            )
        )
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(
            layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            )
        )
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
            )
        )
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )


    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator, epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator, epochs, self.seed)


    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")

        plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
        plt.show()

    def restore_latest(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)) 


shapegan = shapeGAN(batch_size=128)
# https://keras.io/api/preprocessing/image/
# dataset = keras.preprocessing.image_dataset_from_directory(
#     "path/to/img/directory", batch_size=128, image_size=(128, 128)
# )
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(100).batch(128)
shapegan.train(train_dataset, 1000)
