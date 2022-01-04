#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Read train dataset
train = pd.read_csv('data/train.csv')
X_train = np.array((train.iloc[:, :-1])).reshape(len(train),2)
y_train = np.array(train.iloc[:, -1:]).reshape(len(train))


# Fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# save model
with open("classification_app/linear_regression.model", "wb") as f:
    pickle.dump(model, f)

# To do:
# 1. Save the model.
# 2. Deploy the trained model on a web server.
# 3. Send prediction request (test.csv) to the server and retrieve the results.
# Note: You can use any technology, platform, tool (we used pickle, Docker, FastAPI).