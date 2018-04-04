import codecs

import re

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def count_words_statistic(domain, min_freq_number):
    unique_words = 0
    word_freqs = {}

    fin = codecs.open("../preprocessed_data/" + domain + "/train.txt", 'r', 'utf-8')
    for line in fin:
        words = line.split()

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1

    unique_words_more_min = 0
    for k, v in word_freqs.items():
        if word_freqs[k] > min_freq_number:
            unique_words_more_min += 1

    print ('%i unique words, %i unique words more than min' % (unique_words, unique_words_more_min))


count_words_statistic("beer", 100)
