#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all dependencies
import numpy as np 
import re
from collections import Counter
import matplotlib.pyplot as plt


# In[2]:


#Fix randomness for reproducibility
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


# Build One Hot Encoded Matrices 

# In[5]:


#Generate full vocab list from reviews
tokenized_revs = [re.findall(r"[A-Za-z0-9'\-]+", rev.lower()) for rev in reviews]
#vocab = set([w for rev in tokenized_revs for w in rev])
#Uncomment if I want top 5000 words only
toks=[]
for t in tokenized_revs:
      toks.extend(t)
counts=Counter(toks)
so=sorted(counts.items(), key=lambda item: item[1], reverse=True)
so=list(zip(*so))[0]
vocab = so[:10000]


# In[6]:


# Runtime note: this cell may take a little longer to run
#Create one-hot feature matrix (float32 dt is used to reduce memory usage of high-dimensional matrix)
M_base = np.zeros((len(reviews), len(vocab)), dtype=np.float32)
for i, rev in enumerate(reviews):
  rev_tokens = re.findall(r"[A-Za-z0-9'\-]+", rev.lower())
  for j, wor in enumerate(vocab):
    if wor in rev_tokens:
      M_base[i,j] = 1


# In[7]:


#Dimensions sanity check
M_base.shape


# In[8]:


#Save and reload matrix for reuse
#np.save("M_base", M_base)
#M_base = np.load("M_base.npy")


# In[9]:


#Generate random indices for splitting data
train_ints=np.random.choice(len(reviews),int(len(reviews)*0.8),replace=False)
remaining_ints=list(set(range(0,len(reviews))) - set(train_ints))
test_ints = np.random.choice(remaining_ints, int(len(remaining_ints)*0.5), replace=False)
dev_ints = np.array(list(set(remaining_ints) - set(test_ints)))


# In[10]:


#Use train, dev and test integers to create M matrices from embeddings 
base_train = np.array(M_base[train_ints,], dtype=np.float32)
base_dev = np.array(M_base[dev_ints,], dtype=np.float32)
base_test = np.array(M_base[test_ints,], dtype=np.float32)


# In[11]:


#Create binary sentiment labels 
labels = [int(l == "positive") for l in sentiment_ratings]


# In[12]:


#Use train, dev and test integers to create y vectors from labels
y_train = np.array([labels[i] for i in train_ints], dtype = np.float32)
y_dev = np.array([labels[i] for i in dev_ints], dtype = np.float32)
y_test = np.array([labels[i] for i in test_ints], dtype = np.float32)


# Build Logistic Regression Baseline Model

# In[13]:


#Runtime note: this cell may take a little longer to run
#Set model hyperparameters
n_iters = 500
lr = 0.02


num_samples, num_features = base_train.shape
num_classes = 1
weights = np.random.rand(num_features).astype(np.float32)
bias = np.random.rand(1).astype(np.float32)
logistic_loss = []


#Forwards pass through model
for i in range(n_iters):
    z = np.dot(base_train, weights) + bias
    q = 1/(1+np.exp(-z))

    #Calculate loss, using epsilon to avoid errors from log(0)  
    eps=0.00001
    loss = -np.sum((y_train*np.log2(q+eps)+(np.ones(len(y_train))-y_train)*np.log2(np.ones(len(y_train))-q+eps)))
    logistic_loss.append(loss)

    #Calculate partial devs and update weights and bias
    dw = (base_train.T @ (q - y_train)) / num_samples 
    db = sum(q-y_train) / num_samples

    weights -= dw*lr
    bias -= db*lr


# In[14]:


#Inspect loss for fine-tuning hyperparameters
plt.plot(range(1,n_iters),logistic_loss[1:])
plt.xlabel("number of epochs")
plt.ylabel("loss")


# In[15]:


#Save and reload baseline model weights and bias for reuse
#np.save("base_weights", weights)
#weights = np.load("base_weights.npy")


# Evaluating Baseline Model

# In[16]:


#Generate predicted labels for the dev set
z_dev = base_dev.dot(weights) + bias
q_dev = 1/(1+np.exp(-z_dev))
y_dev_pred=[int(ql > 0.5) for ql in q_dev]


# In[17]:


#Generate confusion matrix for evaluation 
dev_t_pos = sum((yp == 1 and y_dev[s] == 1) for s,yp in enumerate(y_dev_pred))
dev_f_pos = sum((yp == 1 and y_dev[s] == 0) for s,yp in enumerate(y_dev_pred))
dev_t_neg = sum((yp == 0 and y_dev[s] == 0) for s,yp in enumerate(y_dev_pred))
dev_f_neg = sum((yp == 0 and y_dev[s] == 1) for s,yp in enumerate(y_dev_pred))


# In[18]:


dev_accuracy = (dev_t_pos + dev_t_neg)/(dev_t_pos + dev_t_neg + dev_f_pos + dev_f_neg)
dev_precision = dev_t_pos/(dev_t_pos + dev_f_pos)
dev_recall = dev_t_pos/(dev_t_pos + dev_f_neg)
print("Evaluation Scores for Dev Set:")
print(f"Acuracy score: {dev_accuracy:.3f}")
print(f"Precision score: {dev_precision:.3f}")
print(f"Recall score: {dev_recall:.3f}")


# In[19]:


#Generate predicted values for test set
z_test = base_test.dot(weights) + bias
q_test = 1/(1+np.exp(-z_test))
y_test_pred = [int (q2 > 0.5) for q2 in q_test]


# In[20]:


#Generate and display confusion matrix for test set
test_t_pos = sum((yp == 1 and y_test[s] == 1) for s,yp in enumerate(y_test_pred))
test_f_pos = sum((yp == 1 and y_test[s] == 0) for s,yp in enumerate(y_test_pred))
test_t_neg = sum((yp == 0 and y_test[s] == 0) for s,yp in enumerate(y_test_pred))
test_f_neg = sum((yp == 0 and y_test[s] == 1) for s,yp in enumerate(y_test_pred))


# In[21]:


#Calculate test accuracy, precision and recall
test_accuracy = (test_t_pos + test_t_neg)/(test_t_pos + test_t_neg + test_f_pos + test_f_neg)
test_precision = test_t_pos/(test_t_pos + test_f_pos)
test_recall = test_t_pos/(test_t_pos + test_f_neg)
print("Evaluation Scores for Test Set:")
print(f"The score for accuracy is: {test_accuracy:.3f}")
print(f"The score for precision is: {test_precision:.3f}")
print(f"The score for recall is: {test_recall:.3f}")


# Inspect Baseline Model Weights

# In[22]:


#Inspect 'negative' weights
[vocab[x] for x in np.argsort(weights)[0:25]]


# In[23]:


#Inspect 'positive' weights
[vocab[x] for x in np.argsort(weights)[::-1][0:25]]

