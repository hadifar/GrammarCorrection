# Grammar correction with neural network

Seq2Seq (Encoder-Decoder) wiht Attention Mechanism for Grammar Correction in Keras.

# How its work
At first, we should create our parallel dataset for training our model. In preprocess folder, lang8 and nucle modules convert each dataset into proper format. Lang8 dataset is very noisy, so I decided to do small preprocessing on that. I remove non-ascii characters, reduce length of character with 3 or more with 1 (e.g token like `!!!!!!!` convert to `!`), and remove unnecessary punctuation (all puntuation except `{',','.','-'}`).

At the final preprocessing step, I do some data augmentation. In each pair (source, target), in addition to existing error, I inject some typo/grmmatical error into the source samples. Things I do in this step include:
 
 - Dropout token
 - Modal replacement
 - Misspelling tokens
 - Change tense of verbs
 - Change singularity/pluarality of nouns
 - Change preposition
 
Accourding to [this](http://www.comp.nus.edu.sg/~nlp/conll14st/CoNLLST01.pdf) paper, above cases will cover most of the errors in English learner writings. 

In training step, I used famous seq2seq Attention model [here](https://arxiv.org/abs/1409.0473). The best hyper-parameters for seq2seq explored by the team at google in "Massive Exploration of Neural Machine Translation Architectures" paper. I used one layer encoder/decoder to keep things as simple as posible. It can be easily extend to 4 layer encoder/decoder famework (considering regularization and dropout).

# How to use

- git clone https://github.com/hadifar/GrammarCorrection.git
- cd GrammarCorrection
- virtualenv venv
- source venv/bin/activate
- sudo pip2 install -r requirements.txt
- mkdir data
- cd data
- Download [lang8](https://sites.google.com/site/naistlang8corpora/) and [NUCLE](http://www.comp.nus.edu.sg/~nlp/corpora.html) and put them in data folder.
- cd ..
- cd preprocess
- sh preprocess_script.sh
- cd ..
- cd models
- sh train_script.sh
- That's all :)

# How to use in Colab
- Look into the [riminder.ipynb](https://github.com/hadifar/GrammarCorrection/blob/master/riminder.ipynb) in the root directory. (do not forget to put the dataset in your google drive).


# TODO:
  - Use fasttext because it has information about the underlying morphology of words and was empirically found to perform better than initializing
  the network randomly or using word2vec.
  - Use character ngram feature
  - Use language model for checking final output
