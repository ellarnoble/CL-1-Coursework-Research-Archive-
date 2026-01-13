#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all dependencies
import numpy as np 
import re
from collections import Counter
import random
import matplotlib.pyplot as plt


# In[2]:


#Fix randomness for reproducibility
random.seed(10)
np.random.seed(10)


# Download Reviews

# In[3]:


get_ipython().system('curl -O https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt')


# In[4]:


#Sort reviews into lists with labels for classification
reviews=[]
sentiment_ratings=[]
product_types=[]
helpfulness_ratings=[]

with open("Compiled_Reviews.txt", encoding="utf-8") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews.append(fields[0])
        sentiment_ratings.append(fields[1])
        product_types.append(fields[2])
        helpfulness_ratings.append(fields[3])


# Donwload Embeddings

# In[5]:


get_ipython().system('pip install gensim')


# In[6]:


#Runtime note: this cell may take a little longer to run
# Download w2vec pretrained embeddings 
import gensim.downloader as api
w = api.load('word2vec-google-news-300')
embeddings_vocab=[x for x in w.key_to_index.keys()]
w.vectors.shape


# Build Embeddings Input Matrix from Reviews

# In[7]:


#Clean text, retaining apostrophes and hypens which are present in word2vec
#Lower case for embedding look-up consistenecy  
tokenized_revs = [re.findall(r"[A-Za-z0-9'\-]+", rev.lower()) for rev in reviews]


# In[8]:


#Runtime note: this cell may take a little longer to run
embeddings = []

#Look up tokens in embeddings vecs
for tokens in tokenized_revs:
    vecs = [w[t] for t in tokens if t in w] 

    #Average embeddings across each review
    if vecs:
        rev_vec = np.mean(vecs, axis=0)
    #Handle any empty vectors
    else:
        rev_vec = np.zeros(w.vector_size) 

    embeddings.append(rev_vec)

#Stack vertically to create input embeddings matrix
embeddings = np.vstack(embeddings)


# In[9]:


#Save review embeddings for reuse
#np.save("embeddings", embeddings)


# In[10]:


#embeddings = np.load("embeddings.npy")
#Dimensions sanity check
print(embeddings.shape)
print(len(reviews))


# Generate Data Splits: Target Domain, Source Domain Training/Development/Test Sets

# In[11]:


#Generate numpy numeric y labels from sentiment ratings for logistic regression
labels = [int(l == "positive") for l in sentiment_ratings]
labels = np.array(labels)
#Generate numpy product types for slicing 
product_types = np.array(product_types)


# In[12]:


#Set 'target' held-out domain from dataset
target_domain = "electronics"
source_index = []
target_index = []

#Generate source and target domain indexes for slicing 
for i, d in enumerate(product_types):
    if d == target_domain:
        target_index.append(i)
    else:
        source_index.append(i)

#Generate X and y vectors for source and target domain data
X_source = embeddings[source_index]
y_source = labels[source_index]

X_target = embeddings[target_index]
y_target = labels[target_index]

#Generate product type vector alighned with X and y source vectors  
product_type_source = product_types[source_index]


# In[13]:


#Generate random indices for splitting data
train_ints=np.random.choice(len(X_source),int(len(X_source)*0.8),replace=False)
remaining_ints=list(set(range(0,len(X_source))) - set(train_ints))
test_ints = np.random.choice(remaining_ints, int(len(remaining_ints)*0.5), replace=False)
dev_ints = np.array(list(set(remaining_ints) - set(test_ints)))


# In[14]:


#Use integers to create train, dev and test sets from source data 
M_train = np.array(X_source[train_ints,])
M_dev = np.array(X_source[dev_ints,])
M_test = np.array(X_source[test_ints,])


# In[15]:


#Use same integers to create train, dev and test y vectors from source data
y_train = np.array([y_source[i] for i in train_ints])
y_dev = np.array([y_source[i] for i in dev_ints])
y_test = np.array([y_source[i] for i in test_ints])


# In[16]:


#Check sentiment class balance across datasets for bias
print("Class balances:")
print(f"Whole dataset: {(sum(y_source) / len(y_source)):.3f}")
print(f"Training set: {(sum(y_train))/(len(y_train)):.3f}")
print(f"Dev set: {(sum(y_dev))/(len(y_dev)):.3f}")
print(f"Test set: {(sum(y_test))/(len(y_test)):.3f}")


# In[17]:


#Check all source product types are present in each split
product_type_train = [product_type_source[int(i)] for i in train_ints]
product_type_dev = [product_type_source[int(i)] for i in dev_ints]
product_type_test = [product_type_source[int(i)] for i in test_ints]
print(len(set(product_type_train)))
print(len(set(product_type_dev)))
print(len(set(product_type_test)))


# In[18]:


#Create dicts of source product types for each data split
source_prods = Counter(product_type_source)
train_prods = Counter(product_type_train)
dev_prods = Counter(product_type_dev)
test_prods = Counter(product_type_test)


# In[19]:


#Rough evaluation of product type balance across datasets, check min and max percentages
def minmax_percent(counts):
    total = sum(counts.values())
    mn = min(counts.values())
    mx = max(counts.values())
    print(f"min: {mn/total*100:.2f}%",
        f"max: {mx/total*100:.2f}%")

print("Source dataset:")
minmax_percent(source_prods)
print("Training set:")
minmax_percent(train_prods)
print("Dev set:")
minmax_percent(dev_prods)
print("Test set:")
minmax_percent(test_prods)


# In[20]:


#Dimensions sanity check
print(M_train.shape)
print(y_train.shape)


# Building a Logistic Regression Classifier

# In[21]:


#Runtime note: this cell may take a little longer to run
#Set model hyperparameters
n_iters = 3000
lr = 0.06

#Set model dimensions and initialise loss
num_samples, num_features = M_train.shape
num_classes = 1
weights = np.random.rand(num_features)
bias = np.random.rand(1)
logistic_loss = []

#Create batches for batch training
batch_size = 512
#Ensure n_batches is rounded up to whole number
n_batches = (num_samples + batch_size - 1) // batch_size
#Create and shuffle batch assighnment labels 
unshuffled_batches = [i % n_batches for i in range(num_samples)]
random.shuffle(unshuffled_batches)
batch_assignments = np.array(unshuffled_batches)
#For each batch j, collect indices of samples assigned to that batch
batch_indexes = []
for j in range(n_batches):
    index_list = [i for i, b in enumerate(batch_assignments) if b == j]
    batch_indexes.append(np.array(index_list))



#Forwards pass through model
for i in range(n_iters):
    cumulative_loss = 0.0
    for j in range(n_batches):
       index = batch_indexes[j]
       input = M_train[index]
       targets = y_train[index]

       z = np.dot(input, weights) +bias
       q = 1/(1+np.exp(-z))
        
       #Calculate loss, using epsilon to avoid any errors from log(0)  
       eps=0.00001
       loss = -np.sum((targets*np.log2(q+eps)+(np.ones(len(targets))-targets)*np.log2(np.ones(len(targets))-q+eps)))
       #Add each batch loss to culmulative epoch loss
       cumulative_loss+=loss
        
       #Calculate partial devs and update weights and bias
       dw = (input.T @ (q - targets)) / len(targets) 
       db = sum(q-targets) / len(targets)
    
       weights -= dw*lr
       bias -= db*lr
    #Append cumulative loss from each epoch to logistic loss    
    logistic_loss.append(cumulative_loss)


# In[22]:


#Save and load model parameters for reuse 
#np.save("source_weights", weights)
#np.save("source_bias", bias)


# In[23]:


#weights = np.load("source_weights.npy")
#bias = np.load("source_bias.npy")


# In[24]:


#Examine loss function over epohcs
plt.plot(range(1,n_iters),logistic_loss[1:])
plt.xlabel("number of epochs")
plt.ylabel("loss")


# Evaluation of the Model on the Dev Set (for fine-tuning hyperparemeters)

# In[25]:


#Generate predicted labels for the dev set
z_dev = M_dev.dot(weights) + bias
q_dev = 1/(1+np.exp(-z_dev))
y_dev_pred=[int(ql > 0.55) for ql in q_dev]


# In[26]:


#Generate confusion matrix for evaluation 
dev_t_pos = sum((yp == 1 and y_dev[s] == 1) for s,yp in enumerate(y_dev_pred))
dev_f_pos = sum((yp == 1 and y_dev[s] == 0) for s,yp in enumerate(y_dev_pred))
dev_t_neg = sum((yp == 0 and y_dev[s] == 0) for s,yp in enumerate(y_dev_pred))
dev_f_neg = sum((yp == 0 and y_dev[s] == 1) for s,yp in enumerate(y_dev_pred))


# In[27]:


#Sanity check
dev_t_pos + dev_f_pos + dev_t_neg + dev_f_neg == len(y_dev)


# In[28]:


dev_accuracy = (dev_t_pos + dev_t_neg)/(dev_t_pos + dev_t_neg + dev_f_pos + dev_f_neg)
dev_precision = dev_t_pos/(dev_t_pos + dev_f_pos)
dev_recall = dev_t_pos/(dev_t_pos + dev_f_neg)
dev_F1 = (2 * (dev_precision * dev_recall)) / (dev_precision + dev_recall)

print("Evaluation Scores for Dev Set:")
print(f"Acuracy score: {dev_accuracy:.3f}")
print(f"Precision score: {dev_precision:.3f}")
print(f"Recall score: {dev_recall:.3f}")
print(f"F1 score: {dev_F1:.3f}")


# Evaluation of Model on the Training Set (to examine possible overfitting)

# In[29]:


#Generate predicted values for training set
z_train = M_train.dot(weights) + bias
q_train = 1/(1+np.exp(-z_train))
y_train_pred = [int (q2 > 0.55) for q2 in q_train]


# In[30]:


#Generate confusion matrix for evaluation
train_t_pos = sum((yp == 1 and y_train[s] == 1) for s,yp in enumerate(y_train_pred))
train_f_pos = sum((yp == 1 and y_train[s] == 0) for s,yp in enumerate(y_train_pred))
train_t_neg = sum((yp == 0 and y_train[s] == 0) for s,yp in enumerate(y_train_pred))
train_f_neg = sum((yp == 0 and y_train[s] == 1) for s,yp in enumerate(y_train_pred))


# In[31]:


train_accuracy = (train_t_pos + train_t_neg)/(train_t_pos + train_t_neg + train_f_pos + train_f_neg)
train_precision = train_t_pos/(train_t_pos + train_f_pos)
train_recall = train_t_pos/(train_t_pos + train_f_neg)
train_F1 = (2 * (train_precision * train_recall)) / (train_precision + train_recall)

print("Evaluation Scores for Training Set:")
print(f"Accuracy score: {train_accuracy:.3f}")
print(f"Precision score: {train_precision:.3f}")
print(f"Recall score: {train_recall:.3f}")
print(f"F1 score is: {train_F1:.3f}")


# Evaluation of Model on the Test Set (for reporting of results)

# In[32]:


#Generate predicted labels for the test set
z_test = M_test.dot(weights) + bias
q_test = 1/(1+np.exp(-z_test))
y_test_pred=[int(ql > 0.55) for ql in q_test]


# In[33]:


#Generate and display confusion matrix for test set
test_t_pos = sum((yp == 1 and y_test[s] == 1) for s,yp in enumerate(y_test_pred))
test_f_pos = sum((yp == 1 and y_test[s] == 0) for s,yp in enumerate(y_test_pred))
test_t_neg = sum((yp == 0 and y_test[s] == 0) for s,yp in enumerate(y_test_pred))
test_f_neg = sum((yp == 0 and y_test[s] == 1) for s,yp in enumerate(y_test_pred))
print("Confusion Matrix:")
print(f"True positives: {test_t_pos}")
print(f"False positives: {test_f_pos}")
print(f"True negatives: {test_t_neg}")
print(f"False negatives: {test_f_neg}")


# In[34]:


test_accuracy = (test_t_pos + test_t_neg)/(test_t_pos + test_t_neg + test_f_pos + test_f_neg)
test_precision = test_t_pos/(test_t_pos + test_f_pos)
test_recall = test_t_pos/(test_t_pos + test_f_neg)
test_F1 = (2 * (test_precision * test_recall)) / (test_precision + test_recall)

print("Evaluation Scores for Test Set:")
print(f"Accuracy score: {test_accuracy:.3f}")
print(f"Precision score: {test_precision:.3f}")
print(f"Recall score: {test_recall:.3f}")
print(f"F1 score: {test_F1:.3f}")


# Evaluating Model Consistency across Product Type Groups

# In[35]:


#Examine precision/recall across each product type to analyse consistency
product_precisions = []
product_recalls = []

for p in set(product_type_test):
    TP = sum((yp == 1 and y_test[i] == 1 and product_type_test[i] == p) for i, yp in enumerate(y_test_pred))
    FP = sum((yp == 1 and y_test[i] == 0 and product_type_test[i] == p) for i, yp in enumerate(y_test_pred))
    TN = sum((yp == 0 and y_test[i] == 0 and product_type_test[i] == p) for i, yp in enumerate(y_test_pred))
    FN = sum((yp == 0 and y_test[i] == 1 and product_type_test[i] == p) for i, yp in enumerate(y_test_pred))

    #Calculate and store precision and recall for each group using eps to avoid division by zero
    product_precision = TP / (TP + FP + eps)
    product_recall    = TP / (TP + FN + eps)
    product_precisions.append(product_precision)
    product_recalls.append(product_recall)

    #Uncomment next line to see full results (excluded for brevity)
    #print(f"Product Type: {p}, Precision: {product_precision:.3f}, Recall: {product_recall:.3f}")


# In[36]:


#Summary statistics across all product type groups 
print("Precision and Recall Statistics across all Product Type Groups:")
print(f"Mean Precision: {np.mean(product_precisions):.3f}")
print(f"Precision SD: {np.std(product_precisions):.3f}")
print(f"Mean Recall: {np.mean(product_recalls):.3f}")
print(f"Recall SD: {np.std(product_recalls):.3f}")


# Examining Dimension Weights and Words

# In[37]:


#Create a vocab from reviews to compare embeddings to weights 
vocab = set([w for rev in tokenized_revs for w in rev])


# In[38]:


#Compute dot product (as a measure of alighnment) between word embeddings and weights 
word_scores = {word: np.dot(weights, w[word]) for word in vocab if word in w}
#Extract words and scores from dictionary for inspection
words = list(word_scores.keys())
scores = np.array(list(word_scores.values()))


# In[39]:


#Inspect 'most positive' words 
top_pos = [words[i] for i in np.argsort(scores)[-25:][::-1]]
for pos in top_pos:
    print(pos)


# In[40]:


#Inspect 'most negative' words
top_neg = [words[i] for i in np.argsort(scores)[:25]]
for neg in top_neg:
    print(neg)


# Evaluating Model on the Target Domain Group

# In[41]:


#Generate predicted labels for the target domain
z_target = X_target.dot(weights) + bias
q_target = 1/(1+np.exp(-z_target))
y_target_pred=[int(ql > 0.55) for ql in q_target]


# In[42]:


#Generate and display confusion matrix for target domain
target_t_pos = sum((yp == 1 and y_target[s] == 1) for s,yp in enumerate(y_target_pred))
target_f_pos = sum((yp == 1 and y_target[s] == 0) for s,yp in enumerate(y_target_pred))
target_t_neg = sum((yp == 0 and y_target[s] == 0) for s,yp in enumerate(y_target_pred))
target_f_neg = sum((yp == 0 and y_target[s] == 1) for s,yp in enumerate(y_target_pred))

print("Confusion Matrix:")
print(f"True positives: {target_t_pos}")
print(f"False positives: {target_f_pos}")
print(f"True negatives: {target_t_neg}")
print(f"False negatives: {target_f_neg}")


# In[43]:


target_accuracy = (target_t_pos + target_t_neg)/(target_t_pos + target_t_neg + target_f_pos + target_f_neg)
target_precision = target_t_pos/(target_t_pos + target_f_pos)
target_recall = target_t_pos/(target_t_pos + target_f_neg)
target_F1 = (2 * (target_precision * target_recall)) / (target_precision + target_recall)


# In[44]:


print("Evaluation Scores for Test Set:")
print(f"Accuracy score: {target_accuracy:.3f}")
print(f"Precision score: {target_precision:.3f}")
print(f"Recall score: {target_recall:.3f}")
print(f"F1 score: {target_F1:.3f}")

