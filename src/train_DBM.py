#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import datetime
import argparse
from obj.DBM import DBM
import tensorflow as tf

################################
# train DBM from MNIST
################################

def trainDBM(learning_rate, k1, k2, k3, epochs, batch_size, dims):
    # import mnist data
    print("importing MNIST training data")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = [tf.reshape(x,shape=(784,1)) for x in x_train]
    x_train = [tf.cast(x,"float32") for x in x_train]
    # create log directory
    current_time = getCurrentTime()
    os.makedirs("pickles/"+current_time)
    # parse string input into integer list
    dims = [int(el) for el in dims.split(",")]
    dbm = DBM(dims, learning_rate, k1, k2, k3, epochs, batch_size)
    dbm.train_PCD(x_train)
    # dump dbm pickle
    file_ls = open("pickles/"+current_time+"/dbm.pickle", "wb")
    pickle.dump(dbm, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
    file_ls.close()
    
def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

####################################
# main command call
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate for stacked RBMs, defaults to 0.01")
    parser.add_argument("--k1", type=int, default=1,
                        help="number of gibbs-sampling steps pre-PCD-k algorithm, defaults to 1")
    parser.add_argument("--k2", type=int, default=5,
                        help="number of gibbs-sampling steps during PCD-k algorithm, defaults to 5")
    parser.add_argument("--k3", type=int, default=5,
                        help="number of gibbs-sampling steps before transferring samples to next model, defaults to 5")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of overall training data passes for each RBM, defaults to 1")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="size of training data batches, defaults to 5")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--dimensions', type=str, 
                               help="consecutive enumeration of visible and hidden layers separated by a comma character, eg. 784,500,500,1000", 
                               required=True)
    args = parser.parse_args()
    # train DBM based on parameters
    trainDBM(args.learning_rate,args.k1,args.k2,args.k3,args.epochs,args.batch_size,args.dimensions)