import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=100, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param hidden_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        

        ## TODO: define your trainable variables and/or layers here. This should include an
        ## embedding component, and any other variables/layers you require.
        self.embedding_matrix = tf.keras.layers.Embedding(self.vocab_size,self.embed_size)
        self.dense_layer_1  = tf.keras.layers.Dense(self.hidden_size,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dense_layer_2 = tf.keras.layers.Dense(self.vocab_size,activation='softmax',use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros")




    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        first_words = inputs[:,0]
        second_words = inputs[:,1]
        embed_1 = self.embedding_matrix(first_words)
        embed_2 = self.embedding_matrix(second_words)
        total_embedded = tf.concat([embed_1,embed_2],axis=1)
        d1 = self.dense_layer_1(total_embedded)
        r1 = self.leaky_relu(d1)
        logits  = self.dense_layer_2(r1)
    
        
        return logits

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.
        (NOTE: you shouldn't need to make any changes to this function).

        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping

        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int32)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


#########################################################################################

def perplexity_calculate(y_true, y_pred):
    #logits = np.argmax(y_pred, axis=1)
    #logits = np.reshape(len(y_true))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred)
    perplexity = tf.exp(tf.reduce_mean(loss))
    return perplexity

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(False)
    acc_metric  = perplexity_calculate

    # Sanity check for the perplexity calculation
    random_pred = tf.Variable(np.array([[0.1, 0.3, 0.5, 0.1], 
                                        [0.4, 0.3, 0.1, 0.2], 
                                        [0.1, 0.7, 0.1, 0.1], 
                                        [0.3, 0.3, 0.2, 0.2]]))  
    random_true = tf.Variable(np.array([2,0,1,3]))
    np.testing.assert_almost_equal(np.mean(acc_metric(random_true, random_pred)), 2.4446151121745054, decimal=4)

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.01), 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )


#########################################################################################

def process_trigram_data(data):
    X = np.array(data[:-1])
    Y = np.array(data[2:])
    X = np.column_stack((X[:-1], X[1:]))
    return X, Y

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...
    train_file_path = '/Users/navyasahay/Desktop/DL/homework-4p-language-models-nsahay2004/data/train.txt'
    test_file_path  = '/Users/navyasahay/Desktop/DL/homework-4p-language-models-nsahay2004/data/test.txt'



    train_data, test_data, vocab = get_data(train_file_path,test_file_path)
    

    

    X0, Y0  = process_trigram_data(train_data)
    X1, Y1  = process_trigram_data(test_data)

    # Sanity Check!
    assert X0.shape[1] == 2
    assert X1.shape[1] == 2
    assert X0.shape[0] == Y0.shape[0]
    assert X1.shape[0] == Y1.shape[0]

    print(X0.shape)

    # TODO: Implement get_text_model to return the model that you want to use. 
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    words = 'speak to this brown deep learning student'.split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab: print(f"{word1} not in vocabulary")
        if word2 not in vocab: print(f"{word2} not in vocabulary")
        else: args.model.generate_sentence(word1, word2, 20, vocab)

if __name__ == '__main__':
    main()
