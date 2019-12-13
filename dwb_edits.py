from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import input
import tensorflow as tf
import coref_model as cm
import util
import os
import pyhocon
import numpy as np

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:
    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)
  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)
  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  return example


#from util.initialize_from_env()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
name = "final"
print("Running experiment: {}".format(name))
config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
print(pyhocon.HOCONConverter.convert(config, "hocon"))


model = cm.CorefModel(config)
session=tf.Session()
model.restore(session)
text = "After making a splash opening a branch of his family's rural bank in Midtown, records show Aaron Johnson stepped down as CEO of Farmers Bank amid allegations of fraud and a cease and desist consent order filed with the FDIC. The consent order, published by the Federal Deposit Insurance Corp. on June 28, details a list of required actions to address alleged unsafe banking practices relating to loan procedures, management, expense reimbursement and pay and capital. The bank neither denied nor admitted wrong doing with the consent order, but was required to add to its board two directors who have no connections to Aaron Johnson, his father, Larry Johnson, or anyone associated with the bank."

print_predictions(make_predictions(text, model))

#from make_predictions
example = create_example(text)
print(example) #This is json format, w/ metadata
tensorized_example = model.tensorize_example(example, is_training=False)
#len(tensorized_example)=12... a lot of things encoded
#tensorized_example[0] is list of sentences.. the actual words
# tensorized_example[1) is tensor shape (3, 44, 300), for (#sentences, #words, embedding_dim)
print(tensorized_example)
feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
#The code pulls out a given number of possible mentions (e.g. 50 total possible mentions)
    # The starting and ending indices for each mention are in mention_starts / mention_ends
    # e.g. both mention_starts and mention_ends could be len 50
    # The possible mentions frequently overlap each other at this stage
# If there's 50 possible mentions, antecedent_scores would be a 51x51 matrix, w/ scores for each possible pair
    # e.g. antecedent_scores[50,:] is the likelihoods of all previous positions paired with position 51
a, b, c, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)
predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)
example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)

