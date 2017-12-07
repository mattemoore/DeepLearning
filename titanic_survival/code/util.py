from datetime import datetime as dt
import pickle

def print_results(estimator, start_time, end_time, grid, features):
    print('Finished in:', ((end_time - start_time).seconds) // 60, 'minutes.')
    print('Best score:', grid.best_score_)
    print('Best params:', grid.best_params_)
    try:
        for name, score in zip(features, grid.best_estimator_.feature_importances_):
            print('Feature importance:', name, '\t\t', score)
    except AttributeError:
        pass
    # print('All results:', grid.cv_results_)
    # print('Predictions on test set:', grid.predict(test))
    print('========================\n========================')
    

def save_grid(grid):
    filename = '../models/' + str(grid.best_score_) + '-' + dt.now().strftime('%y-%m-%d-%H-%M-%S') + '.pkl'
    with open(filename, 'wb+') as file:
        pickle.dump(grid, file)