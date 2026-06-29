"""
Tutorial can be found here: https://www.youtube.com/watch?v=i_LwzRVP7bg
Reference google collab here: https://colab.research.google.com/drive/1m3oQ9b0oYOT-DXEy0JCdgWPLGllHMb4V?usp=sharing
Dataset here: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
"""
import copy
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf

# =====================================
# --- 1. Dataset ---
# =====================================
pd.set_option("display.max_columns", None)

headers = ['BikeCount', 'Hour', 'Temp', 'Humidity',
           'WindSpeed', 'Visibility', 'DewPointTemp', 'SolarRadiation', 'Rainfall', 'Snow', 'Functional']
df = pd.read_csv("../datasets/seoul+bike+sharing+demand/SeoulBikeData.csv")
df = df.drop(['Date', 'Seasons', 'Holiday'], axis=1)
df.columns = headers

# Convert Functional from words to numbers
df['Functional'] = (df['Functional'] == "Yes").astype(int)

# Make the dataframe only consist of data at Hour 12
df = df[df['Hour'] == 12]

# Now we drop the column
df = df.drop('Hour', axis=1)
# print(df.head())

# =====================================
# --- 2. Visualize the dataset ---
# =====================================
features = df.columns[1:]
number_of_features = len(features)

chosen_columns = 3
chosen_rows = math.ceil(number_of_features/chosen_columns)

fig, axes = plt.subplots(chosen_rows, chosen_columns, figsize=(5*chosen_rows, 4*chosen_columns))

# Change to 1D array
axes = axes.flatten()

for i, label in enumerate(features):
    ax = axes[i]
    ax.scatter(df[label], df['BikeCount'])
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel('Bike Count at Noon')

# Hide any unused plots
for j in range(len(features)+1, len(axes)):
    fig.delaxes(axes[j])

# Automatically adjust spacing
plt.tight_layout()

# Save Plot
plt.savefig("../datasets/seoul+bike+sharing+demand/SeoulBikeData_DataPlot.png")

# =====================================
# --- 3. Looking at the dataset, we can drop unneeded features ---
# =====================================
df = df.drop(['Rainfall', 'Snow', 'Functional'], axis=1)
print(df.head())

# =====================================
# --- 4. Split Train/Test/Validation Dataset ---
# =====================================
def get_xy(dataframe, y_label, x_labels=None):
    """
    Split the dataframe into the features and the results
    :param dataframe: The dataframe
    :param y_label: The results
    :param x_labels: The features in a list
    :return:
        data, X, y
    """
    dataframe = dataframe.copy()
    if x_labels is None:
        X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels) == 1:
            X = dataframe[x_labels[0]].values.reshape(-1,1)
        else:
            X = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1,1)
    data = np.hstack((X,y))

    return data, X, y

# =====================================
# --- 5. Doing a training with all features ---
# =====================================
sample_df = df.sample(frac=1)

train_amt = int(0.6*len(sample_df))
valid_amt = int(0.8*len(sample_df))

train = sample_df.iloc[:train_amt]
val = sample_df.iloc[train_amt:valid_amt]
test = sample_df.iloc[:valid_amt]

_, X_train_temp, y_train_temp = get_xy(train, "BikeCount", x_labels=['Temp'])
_, X_val_temp, y_val_temp = get_xy(val, "BikeCount", x_labels=['Temp'])
_, X_test_temp, y_test_temp = get_xy(test, "BikeCount", x_labels=['Temp'])

_, X_train_all, y_train_all = get_xy(train, "BikeCount", x_labels=df.columns[1:])
_, X_val_all, y_val_all = get_xy(val, "BikeCount", x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, "BikeCount", x_labels=df.columns[1:])

temp_regression = LinearRegression()
temp_regression.fit(X_train_all, y_train_all)
print(temp_regression.score(X_test_all, y_test_all))

# =====================================
# --- 5. Doing a training with Neural Network Regression ---
# =====================================
def plot_loss(history):
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()

temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])

temp_nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

history = temp_nn_model.fit(
    X_train_temp.reshape(-1), y_train_temp,
    verbose=0,
    epochs=1000,
    validation_data=(X_val_temp, y_val_temp)
)

plot_loss(history)

