import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg index->word mapping)
    """
    # Hint: You might not use all of the initialized variables depending on how you implement preprocessing. 
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []




    ## TODO: Implement pre-processing for the data files. See notebook for help on this.
    train_read = open(train_file, 'r')
    train_lines = train_read.readlines()
    test_read = open(test_file, 'r')
    test_lines = test_read.readlines()

    
    

    for tr_l in train_lines:
        train_data.extend(tr_l.split())
    
    
    for t_l in test_lines:
       test_data.extend(t_l.split())
    
    train_data = [t.lower() for t in train_data]
    test_data = [t.lower() for t in test_data]


    
    unique_tokens = np.unique(train_data)

    count = 0 

    for u in unique_tokens:
        vocabulary[u] = count
        count += 1 

    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    
    # Uncomment the sanity check below if you end up using vocab_size
    # Sanity check, make sure that all values are withi vocab size
    # assert all(0 <= value < vocab_size for value in vocabulary.values())

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print("train_data", train_data)
    return train_data, test_data, vocabulary
