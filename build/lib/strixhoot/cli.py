import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from .trainer import main
from .data_preprocessor import DataPreprocessor
from .utils import load_model_artifacts, create_directory, set_random_seeds

def main():
    parser = argparse.ArgumentParser(
        description="VAE-GAN Boost: Hybrid Generative Modeling with LightGBM Integration"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Training command
    train_parser = subparsers.add_parser('train', help='Train hybrid model')
    train_parser.add_argument('input', type=str, help='Path to input CSV file')
    train_parser.add_argument('--target', required=True, type=str, 
                            help='Name of target column')
    train_parser.add_argument('--features', nargs='+', required=True,
                            help='List of feature column names')
    train_parser.add_argument('-o', '--output-dir', default='models',
                            help='Output directory for trained models')
    train_parser.add_argument('--iterations', type=int, default=5,
                            help='Number of training iterations')
    train_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')

    # Generation command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic samples')
    gen_parser.add_argument('model_dir', type=str, 
                          help='Path to trained model directory')
    gen_parser.add_argument('--class-label', type=int, required=True,
                          help='Class label to generate samples for')
    gen_parser.add_argument('--num-samples', type=int, default=100,
                          help='Number of samples to generate')
    gen_parser.add_argument('-o', '--output', required=True,
                          help='Output CSV file path')

    # Prediction command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('input', type=str, 
                           help='Path to input CSV file for prediction')
    pred_parser.add_argument('model_dir', type=str,
                           help='Path to trained model directory')
    pred_parser.add_argument('-o', '--output', required=True,
                           help='Output CSV file path for predictions')

    args = parser.parse_args()

    set_random_seeds(args.seed if 'seed' in args else 42)

    if args.command == 'train':
        train_model(args)
    elif args.command == 'generate':
        generate_samples(args)
    elif args.command == 'predict':
        make_predictions(args)

def train_model(args):
    """Handle training command"""
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        df, args.features, args.target
    )
    
    trainer = HybridModelTrainer(config={
        'model_dir': args.output_dir,
        'num_classes': len(np.unique(y_train))
    })
    
    print(f"Starting training with {args.iterations} iterations...")
    trainer.training_loop(
        X_train, y_train,
        X_test, y_test,
        iterations=args.iterations
    )
    
    print(f"\nTraining complete! Best models saved to {args.output_dir}")

def generate_samples(args):
    """Handle sample generation command"""
    model_dir = Path(args.model_dir)
    
    # Load CVAE and CGAN models
    cvae = load_model_artifacts(model_dir/'cvae'/'best_cvae.keras')['model']
    cgan = load_model_artifacts(model_dir/'cgan'/'generator.h5')['model']
    
    # Generate from both models
    cvae_samples = cvae.generate(args.class_label, args.num_samples)
    cgan_samples = cgan.generate_samples(args.class_label, args.num_samples)
    
    # Combine and save samples
    combined_samples = np.vstack([cvae_samples, cgan_samples])
    df = pd.DataFrame(combined_samples, columns=[f"feature_{i}" for i in range(combined_samples.shape[1])])
    df['class_label'] = args.class_label
    
    create_directory(Path(args.output).parent)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(combined_samples)} samples saved to {args.output}")

def make_predictions(args):
    """Handle prediction command"""
    # Load preprocessing and model artifacts
    scaler = joblib.load(Path(args.model_dir)/'scaler.pkl')
    lgb_model = load_model_artifacts(Path(args.model_dir)/'lgbm'/'best_lgbm.pkl')['model']
    
    # Load and preprocess data
    df = pd.read_csv(args.input)
    X = scaler.transform(df.values)
    
    # Make predictions
    predictions = lgb_model.predict(X)
    
    # Save results
    df['prediction'] = predictions
    create_directory(Path(args.output).parent)
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()