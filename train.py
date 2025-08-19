"""
Command-line entry point for training the Transformer News Classification model.

This script serves as the main executable for initiating the model training
process. It is designed to be called directly from the terminal.

Its primary responsibilities are:
1.  Parsing command-line arguments to configure the training run (e.g.,
    specifying dataset size and number of epochs).
2.  Providing clear feedback to the user about the chosen configuration.
3.  Calling the core training logic, which is encapsulated within the
    `transformer_news` package.

This separation of concerns (a simple entry point script vs. a complex library
package) is a robust software design pattern.

Usage:
    To run a quick training session on a small sample of the data for debugging
    or rapid iteration:
    $ python train.py --use-sample --num-epochs 5 --sample-size 2000

    To run a full, definitive training session on the entire dataset:
    $ python train.py --num-epochs 20

Command-line Arguments:
    --use-sample (flag):
        If this flag is present, the script will use a small subset of the
        dataset for training and evaluation. If omitted, the full dataset
        is used by default.
    --num-epochs (int):
        The maximum number of epochs to train the model for. Defaults to 10.
    --sample-size (int):
        Specifies the number of records to use if `--use-sample` is active.
        Defaults to 1000.
"""
import argparse
import logging

from transformer_news import training_engine, config

# Basic logging configuration for the script's feedback
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run Transformer News Classification Training.",
        formatter_class=argparse.RawTextHelpFormatter # Improves help message formatting
    )

    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use a small sample of the dataset instead of the full dataset.\nIf omitted, the full dataset will be used."
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="The maximum number of training epochs. (Default: 10)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="The number of data samples to use if --use-sample is active.\n(Default: 1000)"
    )
    args = parser.parse_args()

    # The logic is now inverted: full_dataset is True if --use-sample is NOT present.
    # This makes the default behavior "use full dataset," which is safer for production runs.
    full_dataset = not args.use_sample

    # --- User Feedback ---
    logging.info("▶ Training run initiated with the following configuration:")
    logging.info(f"   - Full Dataset Used: {full_dataset}")
    if not full_dataset:
        logging.info(f"   - Sample Size:       {args.sample_size} records")
    logging.info(f"   - Max Epochs:        {args.num_epochs}")
    logging.info(f"   - Model Save Path:   {config.MODEL_SAVE_PATH}")
    logging.info("-" * 40)

    # --- Execute Core Logic ---
    # Call the main function from the package, passing the parsed arguments.
    training_engine.main(
        full_dataset=full_dataset,
        num_epochs=args.num_epochs,
        sample_size=args.sample_size
    )

    logging.info("✅ Training run script finished.")
    