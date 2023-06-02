import pandas as pd

file_path = '../data/Course_Summary_2022_2023.json'
df = pd.read_json(file_path)
print(df.head())
from sklearn.model_selection import train_test_split

X = df['Cap'].values.reshape(-1, 1)
y = df['Enrolled'].values

# Split the data into training and validation sets (70% validation, 30% training)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.7, random_state=42)

from sklearn.linear_model import LinearRegression


model = LinearRegression()


model.fit(X_train, y_train)


score = model.score(X_valid, y_valid)

print("R-squared score: {:.2f}".format(score))
