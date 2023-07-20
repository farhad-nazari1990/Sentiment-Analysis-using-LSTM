Sentiment Analysis using LSTM

This repository contains code for a Sentiment Analysis model using LSTM (Long Short-Term Memory) neural networks. The model is trained to classify text into different sentiment categories based on a given dataset. The dataset used for this project is the "BBC News Summary" dataset, which contains news articles and their corresponding categories.
Requirements

To run the code, you'll need the following dependencies:

    Python 3.x
    TensorFlow 2.x
    Numpy
    Matplotlib
    Scikit-learn

You can install the required libraries using pip:

bash

pip install tensorflow numpy matplotlib scikit-learn

Dataset

The dataset used for this project is the "BBC News Summary" dataset, which contains news articles along with their corresponding categories. The dataset is available in CSV format, and we preprocess the text data by tokenizing, padding, and converting text to numerical sequences using the Tokenizer class provided by Keras.
Code Structure

The main components of the code are as follows:

    data_preprocessing.py: This script contains functions to load and preprocess the dataset. It tokenizes the text data, creates padded sequences, and converts labels to numerical sequences.

    sentiment_analysis_lstm.py: This script defines the LSTM-based model using Keras. The model architecture includes an embedding layer, LSTM layers with dropout and batch normalization, and fully connected layers for classification.

    train_model.py: This script loads the preprocessed data, creates the LSTM model, and trains the model using early stopping to avoid overfitting.

    evaluation.py: This script evaluates the trained model on both the training and validation sets. It calculates accuracy, precision, recall, and F1-score metrics.

    visualize_embeddings.py: This script generates word embeddings from the trained model and saves them for visualization.

Training the Model

To train the model, run the train_model.py script. The model will be trained on the preprocessed data using early stopping, which helps prevent overfitting. The training process will be visualized with accuracy and loss plots.
Evaluation

After training, the model's performance will be evaluated on both the training and validation sets using accuracy, precision, recall, and F1-score metrics.
Word Embeddings

The model's word embeddings can be visualized using the visualize_embeddings.py script. The word embeddings will be saved in vecs.tsv and meta.tsv files, which can be uploaded to the TensorFlow Embedding Projector for visualization.
Conclusion

This Sentiment Analysis model using LSTM demonstrates good performance on both the training and validation sets. The implementation includes various techniques, such as dropout, batch normalization, and early stopping, to improve generalization and prevent overfitting. Feel free to experiment with different hyperparameters, model architectures, or datasets to further enhance the model's performance.

Happy coding!
