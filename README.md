# Mixed-data vae

Tensorflow 2 and Tensorflow probability implementation of variational autoencoder(VAE) for mixed multivariate data.

Autoencoders are a combination of two neural networks encoder and decoder. Encoder takes the input data and creates a lower dimensional representation of it. Decoder takes the representation and creates a reconstruction of the original data from it.

\begin{align*}
{\sf z} & = {\sf Encoder(X)} \\
{\sf X_{rec}} & = {\sf Decoder(X)} \\
\end{align*}

Regular autoencoder are used for example in dimensionality reduction tasks. Variational autoencoder differ from normal ones in a sense that the loss function of the model is constructed in a such way, that the representation follows the standard normal distribution.

\begin{align*}
{\sf z} & = {\sf Encoder(X)} \\
{\sf z} &\sim {\sf Norm(0, I)}\\
\end{align*}

Loss function also includes a reconstruction error. Decoder has to learn to create realistic reconstructions. For real valued data one could just choose a proper activation function, that maps data to the domain of the original data. Here we use a completely probabilistic approach and decoder parametrizes a probability distribution for each variable. 

\begin{align*}
{\sf \omega} & = {\sf Decoder(z)} \\
{\sf X_{rec}} &\sim {\sf f(\omega)}\\
\end{align*}

Now one can calculate probabilities for the original data given the newly parametrized distribution. If the probability is high the reconstruction is successful.

\begin{align*}
{\sf Loss} & = {\sf f(\omega, X)} \\
\end{align*}
