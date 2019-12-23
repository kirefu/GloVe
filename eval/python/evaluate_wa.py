#!/usr/bin/env python3.7
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    print('Loading vector file')
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    
    print('vocab')
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    
    print('W')
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v
    print('NORMALISING')
    # normalize each word vector to unit length
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    print('evaluating')
    evaluate_vectors(W_norm, vocab, ivocab)

def evaluate_vectors(W, vocab, ivocab):
    """Evaluate the trained word vectors on a variety of tasks"""

    filename = 'filtered-question-words-fr.txt'
    prefix = '/fs/meili0/faheem/gpanlp/GloVe/eval/question-data'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    with open('%s/%s' % (prefix, filename), 'r') as f:
        full_data = [line.rstrip().split(' ') for line in f]
        full_count += len(full_data)
        data = [x for x in full_data if all(word in vocab for word in x)]

    indices = np.array([[vocab[word] for word in row] for row in data])
    ind1, ind2, ind3, ind4 = indices.T

    predictions = np.zeros((len(indices),))
    num_iter = int(np.ceil(len(indices) / float(split_size)))
    for j in range(num_iter):
        print(j, num_iter)
        subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

        pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
            +  W[ind3[subset], :])
        #cosine similarity if input W has been normalized
        dist = np.dot(W, pred_vec.T)

        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf

        # predicted word index
        predictions[subset] = np.argmax(dist, 0).flatten()

    val = (ind4 == predictions) # correct predictions
    count_tot = count_tot + len(ind1)
    correct_tot = correct_tot + sum(val)


    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


if __name__ == "__main__":
    main()
