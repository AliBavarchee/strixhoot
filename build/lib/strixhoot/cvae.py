import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt

# Import utility function to ensure output directories exist
from .utils import create_directory

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
       Args:
           args (tensor tuple): Mean and log of variance of Q(z|X)
       Returns:
           z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CVAE(Model):
    """
    A simple Conditional Variational Autoencoder.
    
    The encoder receives as input a feature vector (x) and a condition (c),
    and produces a latent representation. The decoder reconstructs x from
    the latent code and condition.
    """
    def __init__(self, input_dim, condition_dim, latent_dim=2, intermediate_dim=64, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Build the encoder
        encoder_inputs = Input(shape=(input_dim,), name="encoder_input")
        condition_input = Input(shape=(condition_dim,), name="condition_input")
        x = layers.Concatenate(name="encoder_concat")([encoder_inputs, condition_input])
        h = layers.Dense(intermediate_dim, activation='relu', name="encoder_dense")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
        z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
        
        self.encoder = Model([encoder_inputs, condition_input], [z_mean, z_log_var, z], name="encoder")
        
        # Build the decoder
        latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
        decoder_condition_input = Input(shape=(condition_dim,), name="decoder_condition_input")
        z_cond = layers.Concatenate(name="decoder_concat")([latent_inputs, decoder_condition_input])
        h_decoded = layers.Dense(intermediate_dim, activation='relu', name="decoder_dense")(z_cond)
        decoder_outputs = layers.Dense(input_dim, activation='sigmoid', name="decoder_output")(h_decoded)
        self.decoder = Model([latent_inputs, decoder_condition_input], decoder_outputs, name="decoder")
        
    def call(self, inputs):
        x, c = inputs
        z_mean, z_log_var, z = self.encoder([x, c])
        x_reconstructed = self.decoder([z, c])
        
        # Compute the KL divergence loss and add it to the model loss.
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)
        return x_reconstructed

def train_cvae(input_path, output_path, latent_dim=2, intermediate_dim=64, batch_size=32, epochs=50):
    """
    Train the Conditional Variational Autoencoder (CVAE).
    
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Directory where the trained model and artifacts will be saved.
        latent_dim (int): Dimension of the latent space.
        intermediate_dim (int): Dimension of the hidden layer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
    """
    # Ensure the output directory exists
    create_directory(output_path)
    
    # Load dataset. Assume CSV has feature columns and a "label" column.
    df = pd.read_csv(input_path)
    feature_columns = [col for col in df.columns if col != "label"]
    X = df[feature_columns].values
    # Use the label as the condition. Reshape to have one column.
    c = df["label"].values.reshape(-1, 1)
    
    # Optionally, normalize X here if necessary.
    input_dim = X.shape[1]
    condition_dim = c.shape[1]  # typically 1
    
    # Initialize and compile the CVAE model.
    cvae = CVAE(input_dim, condition_dim, latent_dim=latent_dim, intermediate_dim=intermediate_dim)
    cvae.compile(optimizer='adam', loss='mse')
    
    # Train the CVAE. The model learns to reconstruct X conditioned on c.
    history = cvae.fit([X, c], X,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2,
                       verbose=1)
    
    # Save the trained model.
    model_save_path = os.path.join(output_path, "cvae_model.keras")
    cvae.save(model_save_path)
    print(f"CVAE model saved to {model_save_path}")
    
    # Plot training and validation loss.
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CVAE Training Loss")
    plt.legend()
    loss_plot_path = os.path.join(output_path, "training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")

# If run as a script, allow command-line training.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Conditional Variational Autoencoder (CVAE)")
    parser.add_argument("--input_path", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", required=True, help="Directory to save the trained CVAE and artifacts")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimension of the latent space")
    parser.add_argument("--intermediate_dim", type=int, default=64, help="Dimension of the intermediate hidden layer")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    args = parser.parse_args()
    train_cvae(args.input_path, args.output_path, args.latent_dim, args.intermediate_dim, args.batch_size, args.epochs)
