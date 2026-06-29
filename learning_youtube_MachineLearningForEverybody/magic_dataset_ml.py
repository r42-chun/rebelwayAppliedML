"""
Tutorial can be found here: https://www.youtube.com/watch?v=i_LwzRVP7bg
Reference google collab here: https://colab.research.google.com/drive/16w3TDn_tAku17mum98EWTmjaLHAJcsk0?usp=sharing#scrollTo=-Vohv6aAzZFK
Dataset here: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# =====================================
# --- 1. Dataset ---
# =====================================
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
datapath = "../datasets/magic_gamma_telescope/magic04.data"
df = pd.read_csv(datapath, names=cols)
pd.set_option("display.max_columns", None)
print(df.head())

# Modify class - "g" to 1
df["class"] = (df["class"]=="g").astype(int)
print(df.head())

# Try plotting the dataset
'''
for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
'''
# =====================================
# --- 2. Create Train, Validation, Test Split ---
# =====================================
# df.sample(frac=1) - Shuffle 100% of the data
sample_df = df.sample(frac=1)

train_amt = int(0.6*len(sample_df))
valid_amt = int(0.8*len(sample_df))

train = sample_df.iloc[:train_amt]
valid = sample_df.iloc[train_amt:valid_amt]
test = sample_df.iloc[valid_amt:]

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# =====================================
# --- 3. KNN ---
# =====================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
prediction = knn_model.predict(X_test)
print("=========== KNN Model ============")
print(classification_report(y_test, prediction))

# =====================================
# --- 4. Naive Baise ---
# =====================================
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
prediction = nb_model.predict(X_test)
print("=========== Naive Bayes ============")
print(classification_report(y_test, prediction))

# =====================================
# --- 5. Logistic Regression ---
# =====================================
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
prediction = lr_model.predict(X_test)
print("=========== Logistic Regression ============")
print(classification_report(y_test, prediction))

# =====================================
# --- 6. SVM ---
# =====================================
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
prediction = lr_model.predict(X_test)
print("=========== Support Vector Classifier ============")
print(classification_report(y_test, prediction))

# =====================================
# --- 7. Neural Net ---
# =====================================
import tensorflow as tf

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs, X_test, y_test):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0
    )

    return nn_model, history

'''
least_val_loss = float('inf')
least_loss_model = None
least_model_hyperparameters = ""
epochs = 100

for num_nodes in [16,32,64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                report = f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}"
                print(report)
                model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs, X_test, y_test)
                plot_history(history)
                val_loss = model.evaluate(X_valid, y_valid)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model
                    least_model_hyperparameters = report

y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

print("=========== NN ============")
print(f"Best Model: {least_model_hyperparameters}")
print(classification_report(y_test, prediction))
'''

# =====================================
# --- 8. Find Best Parameters using Optuna ---
# =====================================
import optuna

def objective(trial):
    # Optuna will intelligently pick values from these lists based on past trials
    num_nodes = trial.suggest_categorical('num_nodes', [16,32,64])
    dropout_prob = trial.suggest_categorical('dropout_prob', [0, 0.2])
    learning_rate = trial.suggest_categorical('lr', [0.01, 0.005, 0.001])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Recreate the model here
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Fit the model
    epochs = 20
    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0
    )

    # Return the metric you want to minimize
    val_loss =  min(history.history['val_loss'])
    print(f"history.history is {type(history.history)}")
    return val_loss

# Create Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# 6. Print the best results
print("Best Hyperparameters found:")
print(study.best_params)
print(f"Lowest Validation Loss: {study.best_value}")