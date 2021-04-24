import os
import numpy as np
import pandas as pd
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import sys
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    file_name = sys.argv[1]
    file_name_2 = sys.argv[2]
    
    encoding_dim = int(sys.argv[3])

    train = pd.read_csv(file_name)
    test = pd.read_csv(file_name_2)

    test_ids = test.iloc[:, 0]
    test = test.iloc[:,1:]
    test_scaled = minmax_scale(test, axis = 0)

    ids = train.iloc[:, 0]
    train = train.iloc[:, 1:]

    train_scaled = minmax_scale(train, axis=0)
    ncol = train_scaled.shape[1]
    nrow = train_scaled.shape[0]
    X_train, X_test, Y_train, Y_test = train_test_split(train_scaled, np.zeros((nrow, 1)), train_size=0.9,
                                                        random_state=seed(2017))

    input_dim = Input(shape=(ncol,))

    # Encoder Layers
    encoded11 = Dense(750, activation='relu')(input_dim)
    # encoded12 = Dense(200, activation='relu')(encoded11)

    # Code Layer
    encoded13 = Dense(encoding_dim, activation=None)(encoded11)

    # Decoder Layers
    # decoded1 = Dense(200, activation='relu')(encoded13)
    decoded2 = Dense(750, activation='relu')(encoded13)
    decoded13 = Dense(ncol, activation='tanh')(decoded2)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs=input_dim, outputs=decoded13)

    # Compile the Model
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(X_train, X_train, nb_epoch=10, batch_size=10, shuffle=False, validation_data=(X_test, X_test))
    encoder = Model(inputs=input_dim, outputs=encoded13)
    encoded_input = Input(shape=(encoding_dim,))

    encoded_train = pd.DataFrame(encoder.predict(train_scaled))
    # encoded_train = encoded_train.add_prefix('component_')
    encoded_train.insert(0, "group_id", ids)
    distance_to_k_centers = encoded_train
    distance_to_k_centers_aslist = []
    idx = 0
    for i in range(distance_to_k_centers.shape[0]):
        for j in range(1, distance_to_k_centers.shape[1]):
            idx += 1
            distance_to_k_centers_aslist.append([idx, distance_to_k_centers.iloc[i,0], "COMPONENT_" + str(distance_to_k_centers.columns[j]),distance_to_k_centers.iloc[i,j], distance_to_k_centers.iloc[i,j]])
    reduced_feat_df = pd.DataFrame.from_records(distance_to_k_centers_aslist, index = None)
    reduced_feat_df.columns=["id", "group_id", "feat", "value", "group_norm"]
    reduced_feat_df.to_csv(file_name.split(".")[0] + "_ffae.csv", index = False)


    encoded_test = pd.DataFrame(encoder.predict(test_scaled))
    # encoded_test = encoded_test.add_prefix('component_')
    encoded_test.insert(0, "group_id", test_ids)
    distance_to_k_centers_aslist = []
    idx = 0
    for i in range(distance_to_k_centers.shape[0]):
        for j in range(1, distance_to_k_centers.shape[1]):
            idx += 1
            distance_to_k_centers_aslist.append([idx, distance_to_k_centers.iloc[i,0], "COMPONENT_" + str(distance_to_k_centers.columns[j]),distance_to_k_centers.iloc[i,j], distance_to_k_centers.iloc[i,j]])
    reduced_feat_df = pd.DataFrame.from_records(distance_to_k_centers_aslist, index = None)
    reduced_feat_df.columns=["id", "group_id", "feat", "value", "group_norm"]
    reduced_feat_df.to_csv(file_name_2.split(".")[0] + "_ffae.csv", index = False)

