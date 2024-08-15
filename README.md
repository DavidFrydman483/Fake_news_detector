# Fake News Detection with Fine-Tuned BERT, DistilBERT, and ALBERT: Performance Comparison

## Overview

In an era where information spreads rapidly across digital platforms, distinguishing between real and fake news has become increasingly challenging yet critically important. Fake news can influence public opinion, misinform decision-making, and erode trust in media. This project explores the application of machine learning to detect fake news, focusing on short text snippets rather than full articles, making the task particularly challenging.

In this project, we fine-tuned several state-of-the-art language models, including BERT, DistilBERT, and ALBERT, alongside an LSTM model that serves as a baseline for comparison. By analyzing and comparing the results of these models, we aim to determine the most effective approach for detecting fake news.

## Models

### 1. Pre-trained Language Models (LLMs)
- **BERT (Base version)**: A transformer-based model widely used in various NLP tasks.
- **DistilBERT**: A smaller and faster variant of BERT, designed for resource efficiency.
- **ALBERT**: A version of BERT with parameter-reduction techniques, intended for faster training and deployment.

All LLMs were imported from Hugging Face and fine-tuned using LoRA (Low-Rank Adaptation), making it possible to adapt these pre-trained models to the fake news detection task efficiently.

### 2. LSTM (Long Short-Term Memory)
An LSTM model was implemented to create a baseline for comparison. Although not as powerful as the transformer-based models, LSTMs have traditionally been effective in sequence-based tasks.

## Dataset

Our dataset for training consists of 280,000 short text samples, each no longer than 40 words. These samples were gathered by merging six different datasets. Below is the breakdown of each dataset used:

1. **[Dataset Name 1]**: [Number of samples]
2. **[Dataset Name 2]**: [Number of samples]
3. **[Dataset Name 3]**: [Number of samples]
4. **[Dataset Name 4]**: [Number of samples]
5. **[Dataset Name 5]**: [Number of samples]
6. **[Dataset Name 6]**: [Number of samples]

The merging of these datasets helped to create a diverse and robust training set, ensuring a wide coverage of fake and real news examples.

## Results

The models were evaluated based on four standard binary classification metrics: accuracy, precision, recall, and F1-score. 

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| BERT        | [Insert] | [Insert]  | [Insert] | [Insert] |
| DistilBERT  | [Insert] | [Insert]  | [Insert] | [Insert] |
| ALBERT      | [Insert] | [Insert]  | [Insert] | [Insert] |
| LSTM        | [Insert] | [Insert]  | [Insert] | [Insert] |

### Confusion Matrices

To further analyze the model performance, confusion matrices for each model are provided below:

#### BERT
![BERT Confusion Matrix](path/to/bert_confusion_matrix.png)

#### DistilBERT
![DistilBERT Confusion Matrix](path/to/distilbert_confusion_matrix.png)

#### ALBERT
![ALBERT Confusion Matrix](path/to/albert_confusion_matrix.png)

#### LSTM
![LSTM Confusion Matrix](path/to/lstm_confusion_matrix.png)

### How to Run

This project was originally developed and executed on a Kaggle notebook. Follow these steps to run the model on Kaggle:

1. **Set Up a Kaggle Notebook:**
   - Log in to your Kaggle account.
   - Navigate to the [Kaggle Notebooks](https://www.kaggle.com/notebooks) section.
   - Click on "New Notebook" to create a fresh notebook.

2. **Clone the Repository:**
   - In your Kaggle notebook, clone this repository by running the following command in a new code cell:
     ```bash
     !git clone https://github.com/DavidFrydman483/Fake_news_detector.git
     ```

3. **Navigate to the Project Directory:**
   - After cloning, navigate to the project directory:
     ```bash
     %cd Fake_news_detector
     ```

4. **Open and Run the Jupyter Notebook:**
   - Locate the `.ipynb` file within the repository and open it in your Kaggle notebook environment.
   - Run each cell sequentially to execute the model.

5. **Adjustments for Local Environment (Optional):**
   - If you plan to run this model in a different environment (e.g., local machine), you might need to adjust file paths, install required dependencies, or modify the environment settings.




## Conclusion


## References
- [Dataset Name 1] - [Link or citation]
- [Dataset Name 2] - [Link or citation]
- [Dataset Name 3] - [Link or citation]
- [Dataset Name 4] - [Link or citation]
- [Dataset Name 5] - [Link or citation]
- [Dataset Name 6] - [Link or citation]
