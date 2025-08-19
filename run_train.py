# /run_train.py
import argparse
import logging
from transformer_news import train, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer News Classification Training.")
    
    # RENAMED and IMPROVED argument
    parser.add_argument(
        "--use-sample",
        action="store_true",  # This is the key change!
        help="Use a small sample of the dataset instead of the full dataset."
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    # The logic is now inverted: full_dataset is True if --use-sample is NOT present
    full_dataset = not args.use_sample

    logging.info("▶ Training run started with configuration:")
    logging.info(f"   Full Dataset   : {full_dataset}")
    if not full_dataset:
        logging.info(f"   Sample Size    : {args.sample_size}")
    logging.info(f"   Num Epochs     : {args.num_epochs}")
    # ... rest of the logging ...

    # Call the main function
    train.main(
        full_dataset=full_dataset, 
        num_epochs=args.num_epochs,
        sample_size=args.sample_size
    )
    