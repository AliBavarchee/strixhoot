import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam

# Import utility to ensure output directories exist
from .utils import save_model_artifacts, load_model_artifacts, create_directory, set_random_seeds

def build_generator(latent_dim, condition_dim, output_dim, hidden_dim=64):
    """
    Build the generator model.
    
    Args:
        latent_dim (int): Dimension of the noise vector.
        condition_dim (int): Dimension of the condition input.
        output_dim (int): Dimension of the output (number of features).
        hidden_dim (int): Number of units in hidden layers.
        
    Returns:
        generator (Model): Keras model representing the generator.
    """
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    condition_input = Input(shape=(condition_dim,), name="condition_input")
    merged = layers.Concatenate(name="gen_concat")([noise_input, condition_input])
    x = layers.Dense(hidden_dim, activation='relu')(merged)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    output = layers.Dense(output_dim, activation='sigmoid')(x)
    generator = Model([noise_input, condition_input], output, name="generator")
    return generator

def build_discriminator(input_dim, condition_dim, hidden_dim=64):
    """
    Build the discriminator model.
    
    Args:
        input_dim (int): Dimension of the feature input.
        condition_dim (int): Dimension of the condition input.
        hidden_dim (int): Number of units in hidden layers.
        
    Returns:
        discriminator (Model): Keras model representing the discriminator.
    """
    feature_input = Input(shape=(input_dim,), name="feature_input")
    condition_input = Input(shape=(condition_dim,), name="condition_input")
    merged = layers.Concatenate(name="disc_concat")([feature_input, condition_input])
    x = layers.Dense(hidden_dim, activation='relu')(merged)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    discriminator = Model([feature_input, condition_input], output, name="discriminator")
    return discriminator

def train_cgan(input_path, output_path, latent_dim=100, hidden_dim=64, batch_size=32, epochs=10000, sample_interval=1000):
    """
    Train a Conditional GAN (CGAN) for regression.
    
    This function loads data from a CSV file (assumes that all columns except "label" are features,
    and "label" is used as the condition), builds the generator and discriminator models,
    and trains the CGAN in an adversarial fashion.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Directory where trained models and artifacts will be saved.
        latent_dim (int): Dimension of the noise vector.
        hidden_dim (int): Hidden layer size for both generator and discriminator.
        batch_size (int): Batch size for training.
        epochs (int): Number of training iterations.
        sample_interval (int): Interval (in epochs) at which to log and sample losses.
    """
    create_directory(output_path)
    
    # Load dataset. Assume all columns except "label" are features.
    df = pd.read_csv(input_path)
    feature_columns = [col for col in df.columns if col != "label"]
    X = df[feature_columns].values.astype('float32')
    # Use "label" column as condition; reshape to (n_samples, 1)
    c = df["label"].values.astype('float32').reshape(-1, 1)
    
    n_samples, output_dim = X.shape
    condition_dim = c.shape[1]
    
    # Build and compile the discriminator
    discriminator = build_discriminator(input_dim=output_dim, condition_dim=condition_dim, hidden_dim=hidden_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    
    # Build the generator
    generator = build_generator(latent_dim=latent_dim, condition_dim=condition_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    
    # Build the combined model
    noise_input = Input(shape=(latent_dim,), name="combined_noise_input")
    condition_input = Input(shape=(condition_dim,), name="combined_condition_input")
    generated_features = generator([noise_input, condition_input])
    # For the combined model, freeze discriminator weights
    discriminator.trainable = False
    validity = discriminator([generated_features, condition_input])
    combined = Model([noise_input, condition_input], validity, name="combined_model")
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    # Labels for real and fake data
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    
    # Arrays to store loss values
    d_losses, g_losses = [], []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random batch of real samples
        idx = np.random.randint(0, n_samples, batch_size)
        real_features = X[idx]
        real_conditions = c[idx]
        
        # Generate a batch of fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_features = generator.predict([noise, real_conditions], verbose=0)
        
        # Train discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch([real_features, real_conditions], real_label)
        d_loss_fake = discriminator.train_on_batch([fake_features, real_conditions], fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # Use the same conditions as the real batch for generator training
        g_loss = combined.train_on_batch([noise, real_conditions], real_label)
        
        # Record losses
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        # Display progress at intervals
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}/{epochs} - D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}% - G loss: {g_loss:.4f}")
    
    # Save the trained generator and discriminator models
    generator_save_path = os.path.join(output_path, "cgan_generator.keras")
    discriminator_save_path = os.path.join(output_path, "cgan_discriminator.keras")
    generator.save(generator_save_path)
    discriminator.save(discriminator_save_path)
    print(f"Generator model saved to {generator_save_path}")
    print(f"Discriminator model saved to {discriminator_save_path}")
    
    # Plot training losses and save the plot
    plt.figure(figsize=(8, 6))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CGAN Training Losses")
    plt.legend()
    loss_plot_path = os.path.join(output_path, "cgan_training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")

# If run as a script, allow command-line training.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Conditional GAN (CGAN) for regression")
    parser.add_argument("--input_path", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", required=True, help="Directory to save the trained CGAN models and artifacts")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the noise vector")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Number of units in hidden layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--sample_interval", type=int, default=1000, help="Interval at which to log and sample losses")
    
    args = parser.parse_args()
    train_cgan(args.input_path, args.output_path, args.latent_dim, args.hidden_dim, args.batch_size, args.epochs, args.sample_interval)
