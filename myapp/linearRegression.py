import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def perform_linear_regression(file_path):
    df = pd.read_json(file_path)

    # Extract features and target variables
    X = df['Cap'].values.reshape(-1, 1)
    y = df['Enrolled'].values

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.7, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Calculate R-squared score on the validation data
    score = model.score(X_valid, y_valid)

    return score

def main():
    file_path = '../data/Course_Summary_2022_2023.json'
    score = perform_linear_regression(file_path)
    print("R-squared score: {:.2f}".format(score))

if __name__ == '__main__':
    main()
