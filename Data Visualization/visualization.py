import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import modules
import sys
sys.path.append('./enrollment_predictions/models')
import course_model as models

def plotData(data):
    data['Offering'] = data['Year'].astype(str) + data['Term']
    sorted_data = data.sort_values('Offering')
    
    courses = sorted_data['Course'].unique()
    sections = sorted_data['Section'].unique()
    
    plt.figure(figsize=(15, 10))

    for course in courses:
        for section in sections:
            section_data = sorted_data[sorted_data['Course'] == course]
            section_data = section_data[section_data['Section'] == section]
            if not section_data.empty:
                plt.plot(section_data['Offering'], section_data['Enrolled'], label=course)
    
    plt.title('Class Enrollment')
    plt.xlabel('Offering')
    plt.ylabel('Number of Students')
    #plt.legend(loc='center left', bbox_to_anchor=(-0.15, 0))
    plt.show()

def plotPredictionsVsActual(model_name, predictions, actual_values):
    plt.figure(figsize=(10, 10))

    # Dotted line
    max_value = max(max(predictions), max(actual_values))
    plt.plot([0, max_value], [0, max_value], 'r--')

    plt.scatter(predictions, actual_values)
    plt.title(f'{model_name} Class Enrollment Predictions vs Actual Values')
    plt.xlabel('Prediction')
    plt.ylabel('Actual Value')
    plt.legend()
    plt.show()

def plotRScores(models, scores):
    plt.figure(figsize=(10, 5))
    plt.bar(models, scores)
    plt.title('R-squared scores')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()

def plotRMSE(models, rmse):
    plt.figure(figsize=(10, 5))
    plt.bar(models, rmse)
    plt.title('RMSE')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()

def main():
    # Plot data
    models.import_data()
    data = models.get_data()
    plotData(data)
    
    # Plot most recent
    score_mr, predictions_mr = models.perform_model("most_recent")
    valid_mr = models.get_y_valid().iloc[:, 0].tolist()
    new_predictions_mr = []
    new_valid_mr = []
    total = 0
    for i in range(len(valid_mr)):
        if predictions_mr[i] != 0:
            new_predictions_mr.append(predictions_mr[i])
            new_valid_mr.append(valid_mr[i])
            total += (predictions_mr[i] - valid_mr[i])**2
    rmse_mr = (total / len(new_valid_mr))**0.5
    plotPredictionsVsActual("Most Recent", new_predictions_mr, new_valid_mr)

    # Plot decision tree
    score_dt, predictions_dt = models.perform_model("decision_tree")
    rmse_dt = models.getRSME(predictions_dt)
    errors_dt = models.getErrors(predictions_dt)
    valid_dt = models.get_y_valid().iloc[:, 0].tolist()
    plotPredictionsVsActual("Decision Tree", predictions_dt, valid_dt)

    # Plot random forest
    score_rf, predictions_rf = models.perform_model("random_forest")
    rmse_rf = models.getRSME(predictions_rf)
    errors_rf = models.getErrors(predictions_rf)
    valid_rf = models.get_y_valid().iloc[:, 0].tolist()
    plotPredictionsVsActual("Random Forest", predictions_rf, valid_rf)

    # Plot linear regression
    score_lr, predictions_lr = models.perform_model("linear_regression")
    rmse_lr = models.getRSME(predictions_lr)
    errors_lr = models.getErrors(predictions_lr)
    valid_lr = models.get_y_valid().iloc[:, 0].tolist()
    plotPredictionsVsActual("Linear Regression", predictions_lr, valid_lr)

    model_names = ["Decision Tree", "Random Forest", "Linear Regression"]
    scores = [score_dt, score_rf, score_lr]
    rmses = [rmse_dt, rmse_rf, rmse_lr]

    plotRScores(model_names, scores)
    plotRMSE(model_names, rmses)

    model_names.append("Most Recent")
    rmses.append(rmse_mr)
    plotRMSE(model_names, rmses)

if __name__ == '__main__':
    main()
