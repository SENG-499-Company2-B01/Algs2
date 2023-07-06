import unittest
import pandas as pd
import numpy as np
import os

from enrollment_predictions.models.auto_regressor_dec_tree import (
    remove_unnecessary_columns,
    feature_engineering,
    prepare_data,
    data_preprocessing,
    train_model,
    predict_year,
    load_enrollment_data
)

test_script_path = os.path.abspath(__file__)
test_script_dir = os.path.dirname(test_script_path)
file_path = os.path.join(test_script_dir, "../../data/client_data/enrollment_data.json")    #this should be changed for new data file
file_path = file_path.replace("\\", "/")

class AutoRegressorDecTreeTests(unittest.TestCase):
    def setUp(self):
        self.data = load_enrollment_data(file_path)
        self.preprocessed_data = data_preprocessing(self.data)

    def test_load_enrollment_data(self):
        test_train_data = load_enrollment_data(file_path)
        expected = self.data.copy()
        if test_train_data is None or test_train_data.empty:
            self.fail("Error: Empty data or failed to load data.")
        self.assertTrue(test_train_data.equals(expected))

    def test_data_preprocessing(self):
        test_preprocessed_data = data_preprocessing(self.data.copy())
        expected_columns_subset = ['Term', 'CRN', 'Num']
        output_columns_subset = test_preprocessed_data.columns[:len(expected_columns_subset)]
        self.assertListEqual(list(output_columns_subset), expected_columns_subset)
        #needs modification after predictor makes adoption to new data file

#    def test_feature_engineering(self):
#        window_sizes = [2, 3, 6, 9]
#        output = feature_engineering(self.preprocessed_data.copy(), window_sizes)
#        expected_columns = [
#            'mean_prev_2', 'median_prev_2', 'min_prev_2', 'max_prev_2', 'std_prev_2', 'ewm_2',
#            'mean_prev_3', 'median_prev_3', 'min_prev_3', 'max_prev_3', 'std_prev_3', 'ewm_3',
#            'mean_prev_6', 'median_prev_6', 'min_prev_6', 'max_prev_6', 'std_prev_6', 'ewm_6',
#            'mean_prev_9', 'median_prev_9', 'min_prev_9', 'max_prev_9', 'std_prev_9', 'ewm_9'
#        ]
#        self.assertListEqual(list(output.columns)[-len(expected_columns):], expected_columns)
#        window_size = 2
#        expected_mean_prev_2 = [np.nan, 10.0, 12.5, np.nan, 5.0, 7.5]
#        print(list(output['mean_prev_{}'.format(window_size)]))
#        self.assertEqual(list(output['mean_prev_{}'.format(window_size)]), expected_mean_prev_2)

        # needs to be finished after predictor makes adoption to new data file

    def test_prepare_data(self):
        test_prepared_data = prepare_data(self.preprocessed_data.copy())
        self.assertTrue(True)
        #needs to be finished after predictor makes adoption to new data file

    def test_remove_columns(self):
        columns_to_drop = ['Term', 'CRN', 'Num', 'Title', 'Units', 'Begin', 'End', 'Days', 'Start_Date', 'End_Date',
                           'Bldg', 'Room', 'Sched_Type', 'Course']

        output = remove_unnecessary_columns(feature_engineering(self.preprocessed_data.copy(), [2, 3, 6, 9]).copy())
        for column in columns_to_drop:
            self.assertNotIn(column, output.columns)


    def test_enrollment_prediction(self):
#        if 'Course' in self.preprocessed_data.columns:
#            print("The 'Course' column is present")
#        else:
#            print("The 'Course' column is not present")

        test_X_train, _ = prepare_data(self.preprocessed_data.copy())
        test_model = train_model(self.preprocessed_data.copy())
        predictions = test_model.predict(test_X_train)
        self.assertIsNotNone(predictions)
        print(predictions)

    def test_predict_year(self):
        test_X_train, _ = prepare_data(self.preprocessed_data.copy())
        test_model = train_model(self.preprocessed_data.copy())
        predictions = predict_year(test_model, test_X_train)
        self.assertIsNotNone(predictions)
        print(predictions)














if __name__ == '__main__':
    unittest.main()
