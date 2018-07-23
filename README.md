# Grammar correction with Neural network

Using Seq2Seq wiht Attention Mechanism for Grammar Correction in Keras.

# How to use

- git clone https://github.com/hadifar/GrammarCorrection.git
- cd GrammarCorrection
- mkdir data
- Download [lang8](https://sites.google.com/site/naistlang8corpora/) and [NUCLE](http://www.comp.nus.edu.sg/~nlp/corpora.html) and put them in data folder.
- cd preprocess
- sh preprocess_script.sh
- cd ..
- cd models
- sh train_script.sh
- That's all :)

# How to use in Colab
- Look into the riminder.ipynb in root directory (make sure put dataset in your google drive).


###TODO:
  - Use fasttext because it has information about the underlying morphology of words and was empirically found to perform better than initializing
  the network randomly or using word2vec.
  
  - Use character ngram feature
  
  - Use Keras Tokenizer and sequence padding
