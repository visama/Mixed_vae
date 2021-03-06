# Mixed-data vae

Tensorflow 2 and Tensorflow probability implementation of variational autoencoder(VAE) for mixed multivariate data. I'm planning to add more notebooks during the summer. Topics could be hyperparameter tuning, medical diagnosis representation learning, replicating results from the applied survival analysis book etc.

Autoencoders are a combination of two neural networks encoder and decoder. Encoder takes the input data and creates a lower dimensional representation of it. Decoder takes the representation and creates a reconstruction of the original data from it.

* z = Encoder(X)
* X_rec = Decoder(z)

Regular autoencoder are used for example in dimensionality reduction tasks. Variational autoencoder differ from normal ones in a sense that the loss function of the model is constructed in a such way, that the representation follows the standard normal distribution.

* z ~ N(0,I)

Loss function also includes a reconstruction error. Decoder has to learn to create realistic reconstructions. For real valued data one could just choose a proper activation function, that maps data to the domain of the original data. Here we use a completely probabilistic approach and decoder parametrizes a probability distribution for each variable. 

* theta = Decoder(z)
* X_rec ~ f(theta)

Now one can calculate probabilities for the original data given the newly parametrized distribution. If the probability is high the reconstruction is successful.

* Loss = f(theta, X)

## Notebooks
[Preprocessing](https://visama.github.io/Mixed_vae/Preprocessing.html)

[Fitting the model](https://visama.github.io/Mixed_vae/Fitting_the_model.html)

[Mixed-vae model methods](https://visama.github.io/Mixed_vae/mixed_vae_class.html)