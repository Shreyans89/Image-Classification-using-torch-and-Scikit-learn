import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.set_palette(sns.color_palette("bright", 8))



from pprint import pprint
import time
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,balanced_accuracy_score
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold,RandomizedSearchCV
from get_image_data_tabular import *


from scipy.stats import norm
from pprint import pprint
import time





# setup a parameter grid, other possible params
#
# criterion              {“gini”, “entropy”}, default=”gini”
# max_depth              int, default=None
# min_samples_leaf       int or float, default=1
# max_features           “auto”, “sqrt”, “log2”}, int or float, default=”auto”
# bootstrap              bool, default=True
#

#'randomforestclassifier__ is the component of the pipeline.n_estimators is the hyperparam
# for logreg:logisticregressionclassifier__l1_lambda etc.

SMALL_PARAM_GRID = {
  'randomforestclassifier__n_estimators': [1,5,15,50,100]
}

   
BIG_PARAM_GRID = {
    'randomforestclassifier__n_estimators': [1,5,15,50,100],
    'randomforestclassifier__criterion': ['gini', 'entropy'],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestclassifier__bootstrap': [True, False],
    'randomforestclassifier__min_samples_leaf': [1,2,3,4],
    'randomforestclassifier__min_samples_split': [3,4,5,6] 
}
   

PARAM_DIST= {
    'randomforestclassifier__n_estimators': stats.randint(100,200),
    'randomforestclassifier__criterion': ['gini', 'entropy'],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestclassifier__bootstrap': [True, False],
    'randomforestclassifier__min_samples_leaf': stats.randint(1,5),
    'randomforestclassifier__min_samples_split': stats.randint(3,6)   
    
}





## create grid and random search tuning cv
def grid_tune(X, y,         # Our data as before 
         param_grid,   # Our search parameters
         cv=None,      # A CV strategy of our choice
         n_jobs=None): # number of parallel jobs 

    # as before except not parameterised
    
    """ perform grid search + 5 fold CV- 5 folds for each hyperparam config"""
    pipeline = make_pipeline(
            RobustScaler(),
            RandomForestClassifier(
                random_state=42 # we only set the random state!
            )
        )
    
    grid_search = GridSearchCV(pipeline,
                           param_grid,
                           scoring='f1_micro',
                           cv=cv,
                           n_jobs=n_jobs,
                              verbose=10)
    
    grid_search.fit(X, y)
        
    return grid_search.cv_results_, grid_search.best_index_,pipeline


def rand_tune(X, y, param_dist, n_iter=50,
             cv=None,
             n_jobs=None):
    """ perform random search + 5 fold CV- 5 folds for each hyperparam config and f1 score metric"""
    pipeline = make_pipeline(
            RobustScaler(),
            RandomForestClassifier(random_state=42)
        )
    
    
    #
    # Same as before, except now we use the RandomizedSearchCV
    #
    grid_search = RandomizedSearchCV(pipeline,
                                   param_dist,
                                   n_iter=n_iter,
                                   scoring='f1_micro',
                                   cv=cv,
                                   verbose=10,
                                   n_jobs=n_jobs)
    
    grid_search.fit(X, y)
        
    return grid_search.cv_results_, grid_search.best_index_,pipeline




def run_random_search(features,labels,param_dist=PARAM_DIST):
    tic = time.perf_counter()
    results, best_index,estimator = rand_tune(features,
                                labels,
                                param_dist,
                                
                                #
                                # Here we set the number of random samples we are attempting
                                #
                                n_iter=50,
                                
                                
                                cv=StratifiedKFold(n_splits=3),
                                n_jobs=5)
    toc = time.perf_counter()
    print_some_facts(results, best_index, tic, toc)
    return results,best_index,estimator



def run_grid_search(features,labels,param_grid=SMALL_PARAM_GRID):
    tic = time.perf_counter()
    results, best_index ,estimator= grid_tune(features,
                                labels,
                                param_grid,
                                
                             
                                
                                cv=StratifiedKFold(n_splits=5),
                                n_jobs=5)
    toc = time.perf_counter()
    print_some_facts(results, best_index, tic, toc)
    return results,best_index,estimator





def print_some_facts(results, best_index, tic, toc):
    print("K Fold Tuning Results")
    print(f"Elapsed Time {toc - tic:0.4f} seconds")
    print("Best Index", best_index)
    print("Best Param Set:")
    pprint(results["params"][best_index])
    

def plot_some_results(results):
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    x = np.linspace(0.85, 1.0, 100)

    mu = results['mean_test_score']
    sigma = results['std_test_score']
    n_estimators = [p["randomforestclassifier__n_estimators"] for p in results["params"]]

    lines = []
    for mu, sigma in zip(results['mean_test_score'], results['std_test_score']):
        pdf = norm.pdf(x, mu, sigma)
        line, = ax.plot(x, pdf, alpha=0.6)
        #line, = fig.add_subplot(x, pdf, alpha=0.6)
        ax.axvline(mu, color=line.get_color())
        ax.text(mu, pdf.max(), f"{mu:.3f}", color=line.get_color(), fontsize=14)
        lines.append(line)

    plt.legend(handles=lines, labels=n_estimators, title="n estimators")
    ax.set_title(f"Average F1 Scores")
    fig.savefig('f1_score_n_estimators.png')
    
# Take top N models
def take_n_best(x, n):
    idxs = [xx-1 for xx in x["rank_test_score"][:n]]
    y = { k:np.array(v)[idxs] for k,v in x.items() }
    y["rank_test_score"] = list(range(0, len(idxs)))


    

if __name__=='__main__':
    ## get image features from PT models
    print('FETCHING TRAIN AND VAL FEATURES AND LABELS FROM IMG FOLDER')
    train_features, train_labels = get_img_training_data(Path('imagewang-160')/'train')
    test_features, test_labels = get_img_training_data(Path('imagewang-160')/'val')
    print('SELECTING ESTIMATOR WITH RANDOM SEARCH OVER THE FOLLOWING GRID:')
    pprint(PARAM_DIST)
    results,best_index,estimator=run_random_search(train_features,train_labels)
    top_results, best_index = take_n_best(results, 5)
    plot_some_results(top_results)
    best_estimator=estimator.set_params(**results['params'][best_index])
    best_estimator.fit(train_features,train_labels)
    y_pred=best_estimator.predict(test_features)
    print('Balanced Accuracy Score of best estimator on validation Data:'+str(balanced_accuracy_score(y_pred,test_labels)))
    
    
    
    
    
    
    
    
   



