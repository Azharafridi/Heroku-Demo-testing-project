import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('D:\Azhar\machine learning\ml model deployement\ML_heruko_deployemet_using_flask\hiring1.csv')
print(dataset)

#dataset['experience'].fillna(0, inplace=True)


dataset['test_score'].fillna(dataset['test_score'].mean(), inplace = True)

X = dataset.iloc[:, :3]

# converting words in to integers # Define the conversion function
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12, 'zero': 0, '0': 0}  # Added '0' as string

    if isinstance(word, int):  # Handle integers
        return word
    elif isinstance(word, str):  # Handle strings
        return word_dict.get(word, 0)
    else:
        return 0  # Handle any other types

# Fill NaN values with 'zero'
X['experience'] = X['experience'].fillna('zero', inplace = True)

# Apply the conversion
X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = dataset.iloc[:,-1]

#splitting training and test set
# since we have a very small dataset, we will train our model with all the available data

regressor = LinearRegression()

# fitting model with training data
regressor.fit(X,y)

# Saving the model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))