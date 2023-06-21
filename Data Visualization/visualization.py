import matplotlib.pyplot as plt
import pandas as pd

# Import modules
import sys
sys.path.append('../enrollment_predictions/models')
import course_model as models

def plotData():
    # read data
    data = pd.read_csv('../data/Course_Summary_2022_2023.csv')

    # plot data
    plt.figure(figsize=(10, 5))
    plt.plot(data['Course'], data['Enrollment'])
    plt.title('Class Enrollment')
    plt.xlabel('Course')
    plt.ylabel('Number of Students')
    plt.show()

def plotPredictionsVsActual(predictions, actual_values):
    data = pd.DataFrame({
        'Predictions': predictions,
        'Actual Values': actual_values
    })

    # plot data
    plt.figure(figsize=(10, 5))
    plt.plot(data['Predictions'], label='Predictions')
    plt.plot(data['Actual Values'], label='Actual Values')
    plt.title('Class Enrollment Predictions vs Actual Values')
    plt.xlabel('Course')
    plt.ylabel('Number of Students')
    plt.legend()
    plt.show()

def plotRScores(models, scores):
    plt.figure(figsize=(10, 5))
    plt.plot(models, scores)
    plt.title('R-squared scores')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()

def main():
    # Linear Regression
    file_path = '../data/Course_Summary_2022_2023.json'
    score = perform_linear_regression(file_path)
    print("R-squared score: {:.2f}".format(score))

if __name__ == '__main__':
    main()
