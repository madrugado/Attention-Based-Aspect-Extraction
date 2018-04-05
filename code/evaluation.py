import argparse
import logging
import numpy as np
from time import time
import utils as U
from sklearn.metrics import classification_report
import codecs

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True,
                    help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200,
                    help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50,
                    help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000,
                    help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=15,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15,
                    help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                    help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                    help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                    help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                    help="The weight of orthogonol regularizaiton (default=0.1)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain
# out_dir = '../pre_trained_model/' + args.domain
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
# assert args.domain in {'restaurant', 'beer'}

from keras.preprocessing import sequence
import reader as dataset

###### Get test data #############
vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)

############# Build model architecture, same as the model used for training #########
from model import create_model
import keras.backend as K
from optimizers import get_optimizer

optimizer = get_optimizer(args)


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)


model = create_model(args, overall_maxlen, vocab)

## Load the save model parameters
model.load_weights(out_dir + '/model_param')
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])


################ Evaluation ####################################

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []

    if domain == 'restaurant':

        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        print(classification_report(true_label, predict_label,
                                    ['Food', 'Staff', 'Ambience', 'Anecdotes', 'Price', 'Miscellaneous'], digits=3))

    elif domain == 'drugs_cadec':
        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        print(classification_report(true_label, predict_label, digits=3))

    else:
        for line in predict:
            label = line.strip()
            if label == 'smell' or label == 'taste':
                label = 'taste+smell'
            predict_label.append(label)

        for line in true:
            label = line.strip()
            if label == 'smell' or label == 'taste':
                label = 'taste+smell'
            true_label.append(label)

        print(classification_report(true_label, predict_label,
                                    ['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:, -1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(open(test_labels), predict_labels, domain)


## Create a dictionary that map word index to word 
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])

#print(type(train_x))
step = len(train_x) / 20
print(step)
train_x = list(train_x)
for i in range(0, 20):
	start = i * step
	end = i * step + step
	print(start)
	print(end)
	batch_train_x = train_x[start: end]
	att_weights, aspect_probs = test_fn([batch_train_x, 0])

        ######### Topic weight ###################################

	topic_weight_out = codecs.open(out_dir + '/topic_weights', 'w', 'utf-8')
	print('Saving topic weights on test sentences...')
	for probs in aspect_probs:
    		weights_for_sentence = ""
    		for p in probs:
        		weights_for_sentence += str(p) + "\t"
    			weights_for_sentence.strip()
    		topic_weight_out.write(weights_for_sentence + "\n")
#print(aspect_probs)

## Save attention weights on test sentences into a file
att_out = codecs.open(out_dir + '/att_weights', 'w', 'utf-8')
print('Saving attention weights on test sentences...')
for c in range(len(test_x)):
    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i != 0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen - line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')

######################################################
# Uncomment the below part for F scores
######################################################

## cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)

# map for the pre-trained restaurant model (under pre_trained_model/restaurant)
# cluster_map = {0: 'Food', 1: 'Miscellaneous', 2: 'Miscellaneous', 3: 'Food',
#            4: 'Miscellaneous', 5: 'Food', 6:'Price',  7: 'Miscellaneous', 8: 'Staff', 
#            9: 'Food', 10: 'Food', 11: 'Anecdotes', 
#            12: 'Ambience', 13: 'Staff'}

cluster_map = {0: 'ADE-Neg', 1: 'ADE-Neg', 2: 'Unknown', 3: 'ADE-Neg', 4: 'ADE-Neg', 5: 'Unknown',
               6: 'Unknown', 7: 'Unknown', 8: 'Unknown', 9: 'ADE-Neg', 10: 'ADE-Neg',
               11: 'Unknown', 12: 'Unknown', 13: 'Unknown', 14: 'Unknown', 15: 'Unknown',
               16: 'Unknown', 17: 'Unknown', 18: 'Unknown', 19: 'Unknown', 20: 'Unknown',
               21: 'Unknown', 22: 'Unknown', 23: 'ADE-Neg', 24: 'ADE-Neg'}

# print '--- Results on %s domain ---' % (args.domain)
# test_labels = '../preprocessed_data/%s/test_label.txt' % (args.domain)
# prediction(test_labels, aspect_probs, cluster_map, domain=args.domain)
