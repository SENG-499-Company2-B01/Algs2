import unittest
import pandas as pd
import numpy as np
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from unittest.mock import patch

from enrollment_predictions.models.auto_regressor_dec_tree import (
    feature_engineering,
    prepare_data,
    data_preprocessing,
    train_model,
    predict_year,
    flatten_data,
    load_enrollment_data,
    calculate_error_metrics
)

test_script_path = os.path.abspath(__file__)
test_script_dir = os.path.dirname(test_script_path)
file_path = os.path.join(test_script_dir, "../../data/client_data/schedules.json")
file_path = file_path.replace("\\", "/")


class AutoRegressorDecTreeTests(unittest.TestCase):
    def setUp(self):
        self.full_data = load_enrollment_data(file_path)
        self.preprocessed_data = data_preprocessing(self.full_data)
        self.sample_data = pd.DataFrame({
            "year": [2008],
            "terms": [
                [
                    {
                        "term": "summer",
                        "courses": [
                            {
                                "course": "CSC100",
                                "sections": [
                                    {
                                        "num": "A01",
                                        "building": "",
                                        "professor": "",
                                        "days": [],
                                        "num_seats": 60,
                                        "enrolled": 18,
                                        "start_time": "",
                                        "end_time": ""
                                    }
                                ]
                            }
                        ]
                    }
                ]
            ]
        })

    def test_load_enrollment_data(self):
        test_train_data = load_enrollment_data(file_path)
        expected = self.full_data.copy()
        if test_train_data is None or test_train_data.empty:
            self.fail("Error: Empty data or failed to load data.")
        self.assertTrue(test_train_data.equals(expected))

    def test_flatten_data(self):
        expected = pd.DataFrame({
            "year": [2008],
            "term": ["summer"],
            "course": ["CSC100"],
            "num_seats": [60],
            "professor": [""],
            "enrolled": [18]
        })
        result = flatten_data(self.sample_data.copy())
        self.assertTrue(result.equals(expected))

    def test_data_preprocessing(self):

        expected = pd.DataFrame({
            "offering": ["CSC100-2008-summer"],
            "year": [2008],
            "term": ["summer"],
            "course": ["CSC100"],
            "num_seats": [60],
            "enrolled": [18],
            "professor": [""]
        })

        result = data_preprocessing(self.sample_data.copy())
        self.assertTrue(result.equals(expected))

    def test_feature_engineering(self):

        input_data = pd.DataFrame({
            "offering": ["CSC100-2008-fall", "CSC100-2008-summer", "CSC100-2009-fall"],
            "year": [2008, 2008, 2009],
            "term": ["fall", "summer", "fall"],
            "course": ["CSC100", "CSC100", "CSC100"],
            "num_seats": [8, 10, 6],
            "enrolled": [241, 18, 300]
        })

        window_sizes = [1, 2]
        result = feature_engineering(input_data, window_sizes)
        result.to_csv("ttt.csv")
        expected = pd.DataFrame({
            'offering': ['CSC100-2008-fall', 'CSC100-2009-fall', 'CSC100-2008-summer'],
            'year': [2008, 2009, 2008],
            'term': ['fall', 'fall', 'summer'],
            'course': ['CSC100', 'CSC100', 'CSC100'],
            'num_seats': [8, 6, 10],
            'enrolled': [241, 300, 18],
            'mean_prev_1': [np.nan, 241, np.nan],
            'median_prev_1': [np.nan, 241, np.nan],
            'min_prev_1': [np.nan, 241, np.nan],
            'max_prev_1': [np.nan, 241, np.nan],
            'std_prev_1': [np.nan, np.nan, np.nan],
            'ewm_1': [np.nan, 241, np.nan],
            'mean_prev_2': [np.nan, np.nan, np.nan],
            'median_prev_2': [np.nan, np.nan, np.nan],
            'min_prev_2': [np.nan, np.nan, np.nan],
            'max_prev_2': [np.nan, np.nan, np.nan],
            'std_prev_2': [np.nan, np.nan, np.nan],
            'ewm_2': [np.nan, 241, np.nan]

        })

        self.assertTrue(result.reset_index(drop=True).equals(expected.reset_index(drop=True)))

    def test_prepare_data(self):
        input_data = pd.DataFrame({
            "offering": ["CSC100-2008-fall", "CSC100-2008-summer", "CSC100-2009-fall"],
            "year": [2008, 2008, 2009],
            "term": ["fall", "summer", "fall"],
            "course": ["CSC100", "CSC100", "CSC100"],
            "num_seats": [8, 10, 6],
            "enrolled": [241, 18, 300]
        })

        result, result_target = prepare_data(input_data.copy())
        result_target.to_csv("ttt.csv")
        expected = pd.DataFrame({
            "year": [2008, 2009, 2008],
            "term": [3, 3, 2],
            "num_seats": [8, 6, 10],
            "mean_prev_1": [241, 241, 241],
            "median_prev_1": [241, 241, 241],
            "min_prev_1": [241, 241, 241],
            "max_prev_1": [241, 241, 241],
            "ewm_1": [241, 241, 241],
            "ewm_2": [241, 241, 241],
            "ewm_3": [241, 241, 241],
            "ewm_6": [241, 241, 241],
            "ewm_9": [241, 241, 241],

        })
        expected = expected.astype(float)

        expected_target = pd.DataFrame({
            "enrolled": [241, 300, 18],
            "offering": ["CSC100-2008-fall", "CSC100-2009-fall", "CSC100-2008-summer"]
        })

        self.assertTrue(result.reset_index(drop=True).equals(expected.reset_index(drop=True)))
        self.assertTrue(result_target.reset_index(drop=True).equals(expected_target.reset_index(drop=True)))

    def test_calculate_error_metrics(self):
        y_true = [10, 20, 30]
        y_pred = [12, 18, 32]

        expected_mae = mean_absolute_error(y_true, y_pred)
        expected_rmse = sqrt(mean_squared_error(y_true, y_pred))
        expected_output = (expected_mae, expected_rmse)

        result = calculate_error_metrics(y_true, y_pred)

        self.assertEqual(result, expected_output)

    def test_predict_yest(self):
        data = data_preprocessing(self.full_data.copy())
        data = data.sort_values('year')

        years = data['year'].unique()
        train_data = data[data['year'] < years[-1]]
        predict_data = data[data['year'] == years[-1]]
        #        featureengineering = feature_engineering(train_data, [1,2])
        #        featureengineering.to_csv('feature.csv', index=False)
        X_train, y_train = prepare_data(train_data)
        X_predict, _ = prepare_data(predict_data)
        model = train_model(X_train, y_train)

        if model is None:
            print("Model training failed.")
            return
        model_features = X_train.columns.tolist()
        result = predict_year(model, predict_data, train_data, model_features)
        self.assertIsNotNone(result)
        print(result)
