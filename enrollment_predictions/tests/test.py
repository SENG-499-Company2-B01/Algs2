import unittest
import pandas as pd
import os
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from enrollment_predictions.models.auto_regressor_dec_tree import *

test_script_path = os.path.abspath(__file__)
test_script_dir = os.path.dirname(test_script_path)
file_path = os.path.join(test_script_dir, "../../data/client_data/schedules.json")
file_path = file_path.replace("\\", "/")


class TestFlattenData(unittest.TestCase):
    def test_flatten_data(self):
        data = pd.DataFrame({
            "year": [2008],
            "terms": [[
                {
                    "term": "summer",
                    "courses": [
                        {
                            "course": "CSC101",
                            "sections": [
                                {"num": "A01", "enrolled": 30},
                                {"num": "A02", "enrolled": 25}
                            ]
                        },
                        {
                            "course": "CSC102",
                            "sections": [
                                {"num": "B01", "enrolled": 20},
                                {"num": "B02", "enrolled": 15}
                            ]
                        }
                    ]
                }
            ]]
        })

        expected_output = pd.DataFrame({
            "year": [2008, 2008],
            "term": ["summer", "summer"],
            "course": ["CSC101", "CSC102"],
            "enrolled": [55, 0]
        })

        actual_output = flatten_data(data)
        pd.testing.assert_frame_equal(actual_output, expected_output)


class TestAutoRegressorFunctions(unittest.TestCase):

    def test_data_preprocessing(self):
        data = load_enrollment_data(file_path)
        data = flatten_data(data)
        processed_data = data_preprocessing(data)

        self.assertIsNotNone(processed_data)

        self.assertTrue(isinstance(processed_data, pd.DataFrame))

        self.assertFalse(processed_data.empty)

        expected_columns = sorted(
            ['year', 'term', 'course', 'enrolled', 'subj_CSC', 'subj_ECE', 'subj_SENG', 'CourseOffering'])
        self.assertListEqual(sorted(list(processed_data.columns)), expected_columns)

        # Check if the returned DataFrame does not contain any NaN values
        self.assertFalse(processed_data.isnull().values.any())


class TestLoadEnrollmentData(unittest.TestCase):
    def setUp(self):
        self.test_file_path = 'test.json'
        self.test_data = {
            "year": [2008],
            "terms": [[
                {
                    "term": "summer",
                    "courses": [
                        {
                            "course": "CSC101",
                            "sections": [
                                {"num": "A01", "enrolled": 30},
                                {"num": "A02", "enrolled": 25}
                            ]
                        }
                    ]
                }
            ]]
        }
        with open(self.test_file_path, 'w') as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_load_enrollment_data(self):
        expected_output = pd.DataFrame(self.test_data)
        actual_output = load_enrollment_data(self.test_file_path)
        pd.testing.assert_frame_equal(actual_output, expected_output)


def ur_model_training_grid_search(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model


class TestModelTrainingGridSearch(unittest.TestCase):
    def test_model_training_grid_search(self):
        X_train = pd.DataFrame({
            "year": [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
            "term": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
            "course": [2101, 2101, 2101, 2101, 2101, 2101, 2101, 2101, 2101, 2101],
            "subj_CSC": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })
        y_train = pd.DataFrame({"enrolled": [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]})

        model = LinearRegression()
        result = ur_model_training_grid_search(X_train, y_train, model)

        self.assertIsInstance(result, LinearRegression)

        self.assertTrue(hasattr(result, 'coef_'))


class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        X_train = pd.DataFrame({
            "year": [2008, 2009],
            "term": [1, 1],
            "course": [2101, 2101],
            "subj_CSC": [1, 1],
        })
        y_train = pd.Series([55, 60])

        model = RandomForestRegressor(random_state=0)
        try:
            model_training(X_train, y_train, model)
        except Exception as e:
            self.fail(f"model_training raised Exception: {e}")






if __name__ == "__main__":
    unittest.main()
