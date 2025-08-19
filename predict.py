"""
Command-line entry point for classifying news headlines.

This script provides a simple and direct way to use the trained Transformer
model to predict the category of a single news headline provided by the user.

It is designed to be called from the terminal and performs the following steps:
1.  Parses a required command-line argument containing the news headline text.
2.  Calls the core prediction logic from the `transformer_news.predict` module.
3.  Prints the final, human-readable prediction to the standard output.

This script is the primary interface for any user or external process wishing
to get a prediction from the model.

Usage:
    To get a prediction for a news headline:
    $ python predict.py --news-article-headline "NASA discovers new planet in a distant galaxy"

    Expected Output:
    Predicted Category: Sci/Tech

Command-line Arguments:
    --news-article-headline (str):
        [Required] The news headline text to be classified, enclosed in quotes.
"""
import argparse
import logging

from transformer_news import predict_engine
from transformer_news import config

if __name__ == "__main__":
    # For a prediction script, logging to the console is often sufficient.
    # Writing to the same training log file might clutter it.
    # If file logging is desired, the original configuration can be used.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()] # Simplified for inference
    )

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run Transformer News Classification Prediction.",
        formatter_class=argparse.RawTextHelpFormatter # Improves help message formatting
    )
    parser.add_argument(
        "--news-article-headline",
        type=str,
        required=True,
        help="The news headline you want to classify (e.g., \"New study finds... C\")."
    )
    args = parser.parse_args()

    # --- Execute Core Logic ---
    # The `main` function from the `predict` module handles all the heavy lifting:
    # loading the model, tokenizing, and running inference.
    try:
        final_prediction = predict_engine.main(news_headline=args.news_article_headline)

        # --- Output Result ---
        # Print the final result to standard output. This makes the script's output
        # easy to capture and use in automated workflows or tests.
        if final_prediction:
            print(f"Predicted Category: {final_prediction} \n")

    except FileNotFoundError:
        logging.error("ERROR: Model or vocabulary file not found.")
        logging.error(f"Please ensure '{config.MODEL_SAVE_PATH}' and '{config.VOCAB_SAVE_PATH}' exist.")
        logging.error("You may need to run the training script first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        