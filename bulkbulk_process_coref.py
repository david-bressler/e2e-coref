


#python2 -m pip install -r requirements.txt


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
import sys
import os
import re
import importlib
import time

#os.chdir('/home/projects/e2e-coref')
#sys.path.append('/home/projects/e2e-coref')
#sys.path.insert(0, '/home/projects/e2e-coref')

#sys.path=['/home/projects/e2e-coref']+sys.path
#print ("\n".join('"'+p+'"' for p in sys.path))

from six.moves import input
import tensorflow as tf
import coref_model as cm
import util
import pyhocon
import numpy as np

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

class Bulk_Coref:
    def __init__(self):
        print("here we go")
        # from util.initialize_from_env()
        gpus = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        name = "final"
        print("Running experiment: {}".format(name))
        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
        config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
        print(pyhocon.HOCONConverter.convert(config, "hocon"))

        self.model = cm.CorefModel(config)
        self.session = tf.Session()
        self.model.restore(self.session)

    def create_example(self,text):
      raw_sentences = sent_tokenize(text)
      sentences = [word_tokenize(s) for s in raw_sentences]
      speakers = [["" for _ in sentence] for sentence in sentences]
      return {
        "doc_key": "nw",
        "clusters": [],
        "sentences": sentences,
        "speakers": speakers,
      }

    def print_predictions(self,example):
      words = util.flatten(example["sentences"])
      for cluster in example["predicted_clusters"]:
        print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

    def make_predictions(self,text):
      example = self.create_example(text)
      tensorized_example = self.model.tensorize_example(example, is_training=False)
      feed_dict = {i:t for i,t in zip(self.model.input_tensors, tensorized_example)}
      _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = self.session.run(self.model.predictions + [self.model.head_scores], feed_dict=feed_dict)
      predicted_antecedents = self.model.get_predicted_antecedents(antecedents, antecedent_scores)
      example["predicted_clusters"], _ = self.model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
      example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
      example["head_scores"] = head_scores.tolist()
      return example

    def normalize_article_dwb(self,body): #based on Nishant's extract_relations.normalize_article
        # HTML Tag
        body = re.sub(r"<br>+", r" ", body)
        # Sign
        body = re.sub(u'\N{COPYRIGHT SIGN}'"|"u'\N{TRADE MARK SIGN}'"|"u'\N{REGISTERED SIGN}', '', body)
        # Whitespaces
        body = re.sub("\s+", " ", body)
        ## Control characters:
        regex = re.compile(r'[\n\r\t]')
        body = regex.sub(" ", body)
        #body = re.sub("\n", ' ', body)
        ## Backslash
        body = re.sub(r"\\",'',body)


        #() defines a group
        # \ means escape the character, e.g. \. means look for actual .
        # [A-Z] means all capital letters
        # (?!\.) is a negative lookaround... i.e. "not ."   (https://www.regular-expressions.info/lookaround.html)
        # r'\1 \2' means put a space between the two backreferred groups defined initially: https://www.regular-expressions.info/backref.html
        body= re.sub('(\.)([A-Z\'\"](?!\.))', r'\1 \2', body) #correct missing space bw period and Capital letter (or quote) starting new sentence
        return body

    def main(self,the_dic):

        for query_num in range(len(the_dic)):
            print("Query Number: " + str(query_num) + " of " + str(len(the_dic)))
            print(the_dic[query_num]["params"]["entity_name"])
            #n_articles=the_dic[query_num]["article_count"]

            f = the_dic[query_num]["results"]
            n_articles=len(f)
            print("Number of Articles: " + str(n_articles))

            for docnum in range(n_articles):

                # text=f[docnum]["body"]
                #print("docnum = " + str(docnum))
                text = f[docnum]["meta"].values()[0]["norm_text"]["body"]
                text = self.normalize_article_dwb(text)
                f[docnum]["preprocessed"] = text
                # print_predictions(make_predictions(text, model))
                result = self.make_predictions(text)
                #self.print_predictions(result)

                f[docnum]["e2e_body"] = util.flatten(result["sentences"])
                f[docnum]["predicted_clusters"] = result["predicted_clusters"]
                clusters_words = []
                for cluster_ind, cluster in enumerate(result["predicted_clusters"]):
                    clusters_words.append([" ".join(f[docnum]["e2e_body"][m[0]:m[1] + 1]) for m in cluster])
                f[docnum]["clusters_words"] = clusters_words
            the_dic[query_num]["results"] = f
        return the_dic



save_filename = '/home/projects/data/bulk_payload_coref.json'

the_dic=[]
for line in open('/home/projects/data/cam_sample_data_payload.json', 'r'):
	# the_dic = json.loads(f.read())
    the_dic.append( json.loads(line))

overall_start_time = time.time()

coreffer = Bulk_Coref()
the_dic = coreffer.main(the_dic)

with open(save_filename, 'w') as outfile:
    json.dump(the_dic, outfile)

overall_end_time = time.time() - overall_start_time
print("OVERALL END TIME: " + str(overall_end_time))