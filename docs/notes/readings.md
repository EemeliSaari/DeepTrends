# Readings

This document contains some of the notes taken from reading topic related articles.

## Articles

### S. N. Kim et al. Automatic keyphrase extraction from scientific articles 2010

**Main:** Define the keyphrase to mean main topic of the document. Can be done ether by unsupervised or supervised approach. Good stuff about pre-processing and NLP on scientific publications in general.

#### Notes

- F-score seem to have been used as measurement. (section 4.)
- Existing datasets are listed in section 3.1
  - Should most likely be looked at.
- Keyphrase extraction is generally construed as a ranking problem.

### C. D. Manning Computational Linguistics and Deep Learning 2015

**Main:** A sort of review article from 2015 ACL. Reviews the affects of deep learning in the computational linguistics field.

#### Notes

- Might not be a very good sitation since seems like it's heavily just personal options.
- Good to find literature though.


### Levy et al. Improving Distributional Similarity with Lessons Learned from Word Embeddings 2015

**Main:** Improvement of the works of Mikolov et al. 2013. Tries to answer why the word2vector is so successful.

#### Notes

- Word2vector still based on the bag-of-context representation of words.
- Literature seems to be focusing on the mathematical focus
  - No deep dive on the hyperparameters
- Hyper parameters seem to have already been fine tuned in original paper but not really explored.
- Shows that hyperparameters can be adapted and transfered into traditional count-based models -> very interesting.

### Kulkarni et al. Statistically Significant Detection of Linguistic Change 2014

**Main:** They track statistically signically significant linguistic shift in the meaning of word  meaning and usage. Analysing the linguistics as time series. Uses variaton of deep learning and and classical methods.

#### Notes

- They train temporal word embeddings for each time "snapshot".
- Allignment of each snapshot into same co-ordinate system.
- Used gensim's skipgram models.
- Great plotting!

### King et al. Computer-Assisted Keyword and Document Set Discovery from Unstructured Text

**Main:** Keywords might be biased and not the optimally chosen -> Tackle this problem with algorithmic approach. Method is a computer-assisted keyword discovery so humans don't have to come up with every relevant keyword.

#### Notes

- Should be very useful reference with chemistry papers evaulation.
- Statistical methodology for keyword detection.

### Yau et al. Clustering Scientific documents with topic modelling

**Main:** Topic modelling for different fields and then cluster them. Explore potential scientometric applications from such methods.

#### Notes

- Uses different variations of LDA (Latent Dirichlet Allocation) for topic modelling.
  - Good material if need to dive deeper to those implementations.
- Uses the topic modelling as dimension reduction to separate different fields from each other.
- Uses the F-Score to evaluate different methods.
- Good assumptions about that should be taken into account ie. should the order of documents matter what topics the model produces?
- Good section on verification dataset on energy technologies (section 5.)
  - Should use the similiar section with chemistry department.

### Hill et al. Learning to Understand Phrases by Embedding the Dictionary

**Main:** Using the dictionaries as source for training context->word type of models (CBOW-type). Uses the recurrent neural network model (RNN) and simple feed forward bag-of-words (BOW) embedding models for the task.

#### Notes

- RNN's with LSTM layers have achieved great success in various NLP tasks such as translation methods.
- Maps the dictionary definition to word embedding that is learned independently with word2vector method.

### Medelyan et al. Automatic construction of lexicons, taxonomies, ontologies, and other knowledge structures 2013

**Main:** Different knowledge structures are important part of natural language understanding. Can be used to ie. build semantic meaning between words automatically.

#### Notes

- Might be a bit off-topic.
- Describe a very abstract ways to implement automated methods.
- Good description of the different knowledge structures.

### Socher et al. Reasoning With Neural Tensor Networks for Knowledge Base Completion

**Main:** 