import unittest
from predict_stocks import StockPredictor
import predict_stocks
import pandas as pd
import numpy as np


class TestStockPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = StockPredictor()

    def tearDown(self):
        pass

    def test_check_missing(self):
        X = pd.DataFrame([[1, 2, ''], [4, 5, 6]])
        self.predictor.check_missing(X)
        self.assertRaises(ValueError, msg='Missing values in data.')

    def test_feature_creator_day_change(self):
        X = pd.DataFrame([[2, 1, 2, 1, 1, 2], [2, 3, 2, 3, 3, 3]])
        creator = self.predictor.FeatureCreator()
        Y = creator.transform(X)
        self.assertEqual(Y[:, predict_stocks.VOL_IDX + 1].sum(), 0)

    def test_add_indicator_feature(self):
        data = pd.DataFrame([[1, 0, 0, 4], [0, 0, 3, 5]],
                            columns=['A', 'B', 'C', 'D'])
        col_to_indicate = 'C'
        new_col, data = \
            self.predictor.add_indicator_feature(data, col_to_indicate)
        self.assertEqual(data.loc[0, new_col], 0)
        self.assertEqual(data.loc[1, new_col], 1)

    def test_split_data(self):
        data = pd.DataFrame(np.arange(100).reshape(-1, 2),
                            columns=['A', 'B'])
        strat_col = 'B'
        data[strat_col] = [0] * 25 + [1] * 25
        train, test = self.predictor.split_data(data, strat_col)
        self.assertEqual(test['B'].mean(), 0.5)

    def test_seperate_targets(self):
        data = pd.DataFrame(np.arange(100).reshape(-1, 4),
                            columns=['A', 'B', 'C', 'D'])
        X_train, X_test = data.iloc[:10, :].copy(), data.iloc[10:, :].copy()
        X_train, Y_train, X_test, Y_test = \
            self.predictor.separate_targets(X_train, X_test, 'D')
        self.assertNotIn('D', X_train.columns)
        self.assertNotIn('D', X_test.columns)
        self.assertEqual(len(X_train) + len(X_test), len(data))
        self.assertEqual(len(Y_train) + len(Y_test), len(data))


if __name__ == '__main__':
    unittest.main()
