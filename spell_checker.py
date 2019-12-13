import json
from wordsegment import segment, load
from nltk import word_tokenize 
load()

with open('words_dictionary.json') as f:
	valid_words = json.load(f)

def correct_spaces(input_exp):
	exp_start = input_exp
	corrected_exp = []
	is_false = 0
	raw_words = []
	list_of_words = word_tokenize(input_exp)
	for word in list_of_words:
		if all([l.isalpha() for l in word]):
			raw_words.append(word)
	for word in raw_words:
		cond = valid_words.get(word.lower())
		if cond is None:
			corrected_word = ' '.join(segment(word))
			for fixed_word in segment(word):
				start_idx = word.lower().find(fixed_word)
				stop_idx = word.lower().find(fixed_word) + len(fixed_word)
				corrected_word = corrected_word.replace(fixed_word, word[start_idx:stop_idx])
			input_exp = input_exp.replace(word, corrected_word)
	exp_end = input_exp
	if not exp_start == exp_end:
		corrected_exp.append(input_exp)
		is_false = 1
	return [is_false, corrected_exp]


#Use the function 'correct_spaces' below this line. Thanks
ex1 = "Danske is under investigation in the United States and several other countries for paymentstotaling 200 billion euros ($223 billion) through its small Estonian branch, many of which the bank said weresuspicious. Together with shipping firm AP Moller-Maersk (MAERSKb.CO) and brewerCarlsberg (CARLb.CO), Danske ispart of a powerful axis in Danish business life and has traditionally been led by either Danish or Scandinavian executives."
ex2 = "Commerzbank, Germany’s second-largest bank behind Deutsche, is looking at possible staff cuts and closing some branches, sources have told Reuters. Danske Chief Executive Chris Vogelzang, who joined Denmark’s largest bank on June 1, sacked former interim CEO Jesper Nielsen that month after thousands of Danish customers were overcharged for an investment product, in his first attempt to draw a line and regain trust from both customers and investors."
ex3 = "My name is Umar Aftab and Iamthebest."
print(correct_spaces(ex3))


