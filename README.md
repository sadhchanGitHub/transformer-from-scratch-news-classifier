# transformer-from-scratch-news-classifier

## A from-scratch implementation of a Transformer Encoder in PyTorch for text classification.

##### This project is a personal learning exercise to build and understand the core components of the Transformer model as introduced in the "Attention Is All You Need" paper. The model is trained on the AG News dataset to classify news articles into one of four categories (World, Sports, Business, Sci/Tech).

# 

##### Architecture

###### The model consists of the following from-scratch PyTorch modules:

###### Multi-Head Self-Attention: Implemented with efficient, vectorized operations for creating Query, Key, and Value projections.

###### Position-wise Feed-Forward Network: The MLP used for deep, non-linear processing of each token.

###### Positional Encoding: Using the sine/cosine formula to inject sequence order information.

###### EncoderLayer: The final assembly combining the above components with residual connections and layer normalization.

# 

##### Results

###### The model was trained on the full AG News dataset (120,000 training samples) using an early stopping mechanism with a patience of 3 to prevent overfitting and save the best-performing model.

###### The best performance was achieved at Epoch 10, reaching a peak Test Accuracy of 91.6%.

###### Peak Test Accuracy: 91.63%

###### Corresponding Train Loss: 0.1486



###### This result demonstrates that the from-scratch implementation is highly effective and successfully learns to generalize from text data.



##### Known Issues \& Future Work

###### While the model achieves a strong overall accuracy of 91.6%, inference testing on specific examples reveals some weaknesses, particularly in distinguishing between Business and Sports news.

# 

##### Observed Failures:

###### An article about the "US economy" was misclassified as "Sports," suggesting the model may be relying on non-topical keywords.

###### A "Premier League" sports article was misclassified as "World," indicating a potential weakness in handling specific proper nouns (entities) that may have been rare in the training data.



##### Proposed Improvements (V2):

###### Subword Tokenization: Using a more advanced tokenizer (like WordPiece or BPE) to better handle rare and out-of-vocabulary words.

###### Pre-trained Embeddings: Leveraging pre-trained embeddings (like GloVe or fastText) to provide the model with a stronger semantic starting point.

###### Advanced Pooling: Using a more sophisticated pooling strategy, like a \[CLS] token, to create the final sentence representation.

###### Hyperparameter Tuning: A more rigorous search over learning rates, model dimensions, and dropout values.



##### How to Run

Clone this repository.

Set up the conda environment and install the required packages.

Run the training\_notebook.ipynb to retrain the model or inference\_notebook.ipynb to make predictions with the pre-trained weights.

