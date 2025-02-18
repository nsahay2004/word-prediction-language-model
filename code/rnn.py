import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        ## TODO:
        ## - Define an embedding component to embed the word indices into a trainable embedding space.
        ## - Define a recurrent component to reason with the sequence of data. 
        ## - You may also want a dense layer near the end...    
        self.embedding_matrix = tf.keras.layers.Embedding(self.vocab_size,self.embed_size)
        self.RNN_component = tf.keras.layers.LSTM(self.rnn_size,return_sequences=True,return_state=False)
        self.dense_layer_1  = tf.keras.layers.Dense(224,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dense_layer_2 = tf.keras.layers.Dense(self.vocab_size,activation='softmax',use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros")


    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """
        embed_1 = self.embedding_matrix(inputs)
        rnn = self.RNN_component(embed_1)
        dense_1 = self.dense_layer_1(rnn)
        relu = self.leaky_relu(dense_1)
        dense_2 = self.dense_layer_2(relu)


        return dense_2

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]])
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            logits = np.array(logits[0,0,:])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array([[out_index]])

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
    model = MyRNN(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(False)
    acc_metric  = perplexity_calculate

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )



#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    train_file_path = '/Users/navyasahay/Desktop/DL/homework-4p-language-models-nsahay2004/data/train.txt'
    test_file_path  = '/Users/navyasahay/Desktop/DL/homework-4p-language-models-nsahay2004/data/test.txt'

    window_size = 5



    train_data, test_data, vocab = get_data(train_file_path,test_file_path)
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    train_remainder = (len(train_data) - 1)%window_size
    test_remainder = (len(test_data) - 1)%window_size


    train_data = train_data[:-train_remainder]
    test_data = test_data[:-test_remainder]


    X0, Y0  = train_data[:-1].reshape(-1, window_size), train_data[1:].reshape(-1, window_size)
    X1, Y1  = test_data[:-1].reshape(-1, window_size), test_data[1:].reshape(-1, window_size)

    ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in 'speak to this brown deep learning student'.split():
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: args.model.generate_sentence(word1, 20, vocab, 10)

if __name__ == '__main__':
    main()
