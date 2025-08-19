# /run_prediction.py

import argparse
import logging
from transformer_news import predict, config

if __name__ == "__main__":
    # Logging setup remains the same
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ],
    )

    # Argparse setup remains the same
    parser = argparse.ArgumentParser(description="Run Transformer News Classification Prediction.")
    parser.add_argument(
        "--news_article_headline",
        type=str,
        required=True,
        help="The news headline you want to classify."
    )
    args = parser.parse_args()

    # STEP 1: Call the main function and CAPTURE its return value
    final_prediction = predict.main(news_headline=args.news_article_headline)

    # STEP 2: PRINT the final result to the console for the test to see
    # This also makes the script more user-friendly for command-line use.
    if final_prediction:
        print(f"Predicted Category: {final_prediction}")