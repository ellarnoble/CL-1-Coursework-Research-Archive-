## CL-1-Coursework-Research-Archive-
# Project Overview 
This repository accompanies a research report investigating sentiment classification across multiple product-type domains using Amazon reviews. The proposed model is designed to be robust and generalisable across a variety of consumer product types. 

# Structure of Repository
The repository is organised into two code files: 
- 'Baseline Model': contains code which was used to generate and evaluate a baseline comparison for the proposed model.
- 'Embeddings Model': contains code which was used to generate and evaluate the model proposed in the research report.

These files can be run independently. The order in which the files are run is up to the preference of the reader.

# Model Task and Design 
The data used in this study is downloaded at the top of each code file and is taken from a publicly available dataset of 36,548 Amazon product reviews which are labelled with sentiment, helpfulness and product type. Both code files contain a logistic regression sentiment classification model. 

The proposed model uses pre-trained 'GoogleNews' word2vec embeddings which are downloaded in the code file. The model is designed for robustness and generalisation to multiple product-type domains. Data splits therefore consist of an evaluation-only 'target domain' (electronics) as well as an 80/10/10 split from the remaining source domain data for training, development and test sets. The success of this task is evaluated by calculating the model's precision, recall, accuracy and F1 score on the held-out source domain test set and the unseen target domain.

# Running the Code
The code was developed and executed using Python 3.12 (64-bit) via Jupyter Notebook in Anaconda Navigator. All code dependencies are imported in the first cell of each file and consist of NumPy, Matplotlib (for visualisation) and selected Python standard library modules.

All code can be run top to bottom. Some cells, in which variables were saved to local machines during development, are commented out to avoid unwanted file creation. Certain cells are marked with a warning for expected additional runtime; however no individual cell should take more than approximately five minutes to run. Running each file will output model and data evaluation metrics as well as a graphical depiction of loss functions. 


# Reproduction of Results
Model evaluation metrics exhibit minor variation across re-training runs using fixed hyperparameters and controlled randomness. This variation is attributed to floating-point arithmetic and does not affect the interpretation of the results. 
