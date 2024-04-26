import re
import pandas as pd
from pprint import pprint
import csv
import numpy as np
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle


def main():
    
    # Open using panda dataframe ###
    nouns = pd.read_csv('nouns.csv')
    
    # Retrieve cores for parallelization
    n_cores = multiprocessing.cpu_count()

    # Extract names and gender
    nouns = nouns[["lemma","genus"]]

    # Drop missing values
    nouns = nouns.dropna()
    
    #nouns = nouns.head(10000)

    # Clean the data to remove anything that starts with numbers 
    # and special characters
    for noun in nouns["lemma"]:
        if re.search(r"^[\W]", str(noun)):
            nouns.drop(nouns[(nouns["lemma"] == noun)].index, inplace=True)

    # Drop missing values after processing
    nouns = nouns.dropna()
    
    # Set names for columns that correspond to the 'n' in n-gram    
    Names = {1: 'Chs', 2: 'Chs2', 3: 'Chs3', 4: 'Chs4', 5: 'Chs5', 6: 'Chs6'}
        
    # Generate n-grams and append each set of n-grams as separate column
    for n in range(1, 6):
        nouns[Names[n]] = nouns['lemma'].apply(lambda i: create_ngram(i, n))

    #print(nouns.head(5))
    
    #print(f"Current shape is {nouns.shape}")
    
    '''
    This creates a few summary statistics for the entire dataset
    
    '''
    
    categories = ['f', 'm', 'n']
    
    genus_counts = nouns['genus'].value_counts(normalize=True)
    print(genus_counts)
    
    '''
    
    This creates a list of dictionaries which contains 
    the total count of n-gram x gender. For instance, it will
    contain how often the n-gram "aa" will occur in each gender.  
    
    '''
    
    lst_absolute_counts = []
    for n in range(1,6):
        absolute_values = create_dictsof_counts(nouns, nouns[Names[n]], 'genus')
        lst_absolute_counts.append(absolute_values.copy())
        absolute_values.clear()
    
    #print (lst_absolute_counts[1])
    
    '''
    
    This converts the list of nested dicts into dataframes we can
    now use to conduct a chi-squared test
    
    '''
    
    absolute_counts_df = convert_dicts_to_df(lst_absolute_counts)
    
    #Redo indexing of the dataframe  
    absolute_counts_df.reset_index(inplace=True)
    absolute_counts_df.rename(columns={'index': 'n-gram'}, inplace=True)
    
    #print(absolute_counts_df.head(1000))

    '''
    
    Conduct chi-square test and return p-value for each row
    
    '''

    absolute_counts_df['P_Value'] = absolute_counts_df.apply(lambda row: calculate_p_value(row, genus_counts), axis=1)  
    
    # Sort the dataframe by p-value
    sorted_absolute_counts_pvalue = absolute_counts_df.sort_values(by = 'P_Value', ascending=True)
    
    '''
    FEATURE REDUCTION
    
    This piece of code looks at the relative frequencies of n-grams and 
    drops those below a certain threshhold    
    
    '''
    # Returns the number of features that are too small in sample size for a chi-squared test
    count_NAN = sorted_absolute_counts_pvalue['P_Value'].isna().sum()
    
    reduced_features = sorted_absolute_counts_pvalue.dropna()
    
    # This returns only the features where the p-value was less than 5%
    reduced_features = reduced_features[reduced_features['P_Value'] < 0.05]
    
    # Reduce features to only those that contain the end character '<E>'
    reduced_features = reduced_features[reduced_features['n-gram'].apply(lambda x: '<E>' in x)]
    reduced_features = reduced_features['n-gram']
    #print(reduced_features.head(50))
    
    print(f"Reduced Feature Length: {len(reduced_features)}")    
    print(reduced_features.head(10))
    
    '''
    
    Construct a dataframe that can be used for model training
    
    '''
    
    decisiontree_df = pd.DataFrame()
    
    # Populate list of words with the start and end symbols
    decisiontree_df['Word'] = '<S>' + nouns['lemma'] + '<E>' 
    
    # Append the gender of each noun to dataframe
    decisiontree_df['Genus'] = nouns['genus']
    
    # Create columns for each feature and build one hot encoding 
    # to indicate if the word contains that n-gram
    for col in reduced_features:
        decisiontree_df[col] = '' 
        decisiontree_df[col] = decisiontree_df['Word'].apply(lambda x: col in x)
    
        
    print(decisiontree_df.head(50))
    
    '''
    
    This block of code implements the decision tree. The dataset is split into 
    three parts: training, test and validation.
    
    '''
    # Pick outcome variable (Y) and features (X)
    Y = decisiontree_df['Genus']
    X = decisiontree_df.drop(columns=['Word' , 'Genus'])
    
    # Split data in training, test and validation using a 3:1:1 split
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, Y, test_size=0.4, random_state=20)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

    
    '''
    
    The following function creates a graph that compares model accuracy 
    to different values of alpha in cost complexity pruning. A value 
    of alpha is chosen by eye-balling.
    
    '''
    
    #choose_ccp_alpha(X_train_temp, y_train_temp, X_test, y_test)
    
    # Use alpha = 0.0001 to create the decision tree and calculate accuracy
    tre = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.0001)
    tre.fit(X_train_temp, y_train_temp)
    y_train_pred = tre.predict(X_train_temp)
    y_test_pred = tre.predict(X_test)
    
    print(f'Post-pruning train score {accuracy_score(y_train_pred,y_train_temp)}')
    print(f'Post-pruning test score {accuracy_score(y_test_pred,y_test)}')
    
    # Analyze tree depth
    depth = tre.get_depth()
    print(f"Tree depth: {depth}")
    
    '''
    
    Print model and feature list for runfile
    
    '''
    
    with open('decision_tree_ccp_model.pkl', 'wb') as file:
        pickle.dump((tre, reduced_features), file)
    

    '''
    
    Visualizes and analyzes the tree creating summary statistics 
    
    '''
    
    #visualize(tre, X, Y, 50)

    
    '''
    
    Prepruning the tree using GridSearch
    
    '''
    
    params = {'max_depth': [*range(10,20)],'min_samples_split': [2,3,4], 'min_samples_leaf': [1,2]}
    
    tre_gcv = run_GSCV_DecisionTree(params, X_train_temp, y_train_temp)
    
    tre_gcv.fit(X_train_temp, y_train_temp)
    y_train_pred = tre_gcv.predict(X_train_temp)
    y_test_pred = tre_gcv.predict(X_test)
        
    print(f'GCV Train score {accuracy_score(y_train_pred,y_train_temp)}')
    print(f'GCV Test score {accuracy_score(y_test_pred,y_test)}')
    
    '''
    
    Random forest classifier
    
    '''
        
    # Initialize Classifier
    rf = RandomForestClassifier()
    
    param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt']
    }
    
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=n_cores)

    
    # Train the Classifier
    random_search.fit(X_train_temp, y_train_temp)
    
    print("Best parameters found: ", random_search.best_params_)
    
    rf_randomsearch = random_search.best_estimator_
    
    # Predict the target value
    
    y_test_pred_rf = rf_randomsearch.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_test_pred_rf)
    
    print("Random forrest accuracy:", accuracy)
    
    '''
    
    Ensemble model combining postpruned Decision tree and Random Forest
    
    '''
     
    ensemble_model = VotingClassifier(estimators=[('model1', tre), ('model2', rf_randomsearch)], voting='hard')
    
    ensemble_model.fit(X_train_temp, y_train_temp)
    
    ensemble_preds = ensemble_model.predict(X_test)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    print("Ensemble Model Accuracy:", ensemble_accuracy)
   
   
''' 

Create a function that builds 2-, 3- and 4-grams from the word,
along with the special characters that indicate the start and 
end of the word

'''

def create_ngram(word, n):
    chr = ['<S>'] + list(str(word)) + ['<E>']

    chs = []
    
    if n == 1: 
        chs = ['<S>'] + list(str(word)) + ['<E>']
        string = chs
    
    else:
        for i in range(0, n):
            
            chs.append(chr[i:])

        chs = list(tuple(zip(*chs)))
        string = [''.join(ch) for ch in chs]

    return string
          
          
'''
Function to conduct chi-square test on a given row and extract p-value

'''  

def calculate_p_value(row, expected_proportions):
    observed = row[['f', 'm', 'n']] 
    expected = expected_proportions * (row['f'] + row['m'] + row['n'])
    
    # Check if all expected counts are above 5, otherwise return NA
    
    if all(count > 5 for count in expected):
        _, p_value, _, _ = chi2_contingency([observed, expected])
        return p_value
    else:
        return np.nan 
    

'''

Function to create decision trees for a give value 
of ccp_aplpha, Xs and Ys 

'''

# For each alpha we will append our model to a list
def train_decision_tree(ccpaplha, xtrain, ytrain):
    dt_classifier = DecisionTreeClassifier(ccp_alpha=ccpaplha, random_state=42)
    dt_classifier.fit(xtrain, ytrain)
    return dt_classifier

'''

Function that creates dicts of counts from...

[TODO]

'''

def create_dictsof_counts(df, col, targetcol):
    absolute_values = {}
    for idx, cell in enumerate(col):
        gender = df.iloc[idx][targetcol]
        for lst in cell:
            absolute_values[lst] = absolute_values.get(lst, {})
            absolute_values[lst][gender] = absolute_values.get(lst, {}).get(gender, 0) + 1
    return absolute_values
  

'''

This function takes a nested dictionary as an input and 
returns a dataframe that can be used for training the model


'''  

def convert_dicts_to_df(list):

    df = pd.DataFrame()
        
    for count, lst in enumerate(list):
        df = pd.concat(
            [df,
            pd.DataFrame.from_dict(list[count], 
                                    orient = 'index')], axis = 0)
    
    return df

'''

This function creates a graph comparing accuracy and values 
of alpha to help choose a value of alpha for cost complexity pruning

'''

def choose_ccp_alpha(X, Y, X_test, y_test):
    
    n_cores = multiprocessing.cpu_count()

    tre = DecisionTreeClassifier()
    
    tre.fit(X, Y)
    
    # Initialize a DataFrame to store results
    results = []
    
    path = tre.cost_complexity_pruning_path(X, Y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = [max(0, x) for x in ccp_alphas]
    #print(f"CCP Alphas have length {len(ccp_alphas)}")
    
    unique_ccps = list(set(ccp_alphas))
    #print(f"CCP Alphas have length {len(unique_ccps)}")
    

    #print("Training trees with ccp-alphas")
    tres = Parallel(n_jobs=n_cores)(
    delayed(train_decision_tree)(ccp_alpha, X, Y) for ccp_alpha in ccp_alphas)
    
    tres = []
    for count, ccp_alpha in enumerate(ccp_alphas):
        print(count)
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X, Y)
        tres.append(clf)
    
    
    
    # Remove the last element in clfs and ccp_alphas and plot trees
    
    tres = tres[:-1]
    ccp_alphas = ccp_alphas[:-1]
    
    # Trying to figure out which alpha to choose by 
    # eyeballing a graph comparing accuracy vs akpha
    
    train_acc = []
    test_acc = []
    for c in tres:
        y_train_pred = c.predict(X)
        y_test_pred = c.predict(X_test)
        train_acc.append(accuracy_score(y_train_pred,Y))
        test_acc.append(accuracy_score(y_test_pred,y_test))

    plt.scatter(ccp_alphas,train_acc)
    plt.scatter(ccp_alphas,test_acc)
    plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
    plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
    plt.legend()
    plt.title('Accuracy vs alpha')
    plt.show()

    
def visualize(tree, features, categories, size):

    # Visualize the tree
    plt.figure(figsize = (size,size))
    plot_tree(tree, filled=True, feature_names=features.columns, class_names=categories.unique())
    plt.show()

    # Analyze feature importance
    print("Feature Importance:")
    for feature, importance in zip(features.columns, tree.feature_importances_):
        print(f"{feature}: {importance}")

    # Analyze node distribution
    print("\nNode Distribution:")
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of leaf nodes: {sum(children_left == -1)}")


    # Analyze leaf node statistics
    leaf_samples = tree.tree_.n_node_samples[children_left == -1]
    class_distribution = tree.tree_.value[children_left == -1]
    print("\nLeaf Node Statistics:")
    for i, (samples, distribution) in enumerate(zip(leaf_samples, class_distribution)):
        print(f"Leaf Node {i}:")
        print(f"  Samples: {samples}")
        print(f"  Class Distribution: {distribution}")

    

'''
Function that runs a Grid Search Decision tree for a set of parameters, 
X = training features, Y = training output 

'''  
    
def run_GSCV_DecisionTree(params, X, Y):
    n_cores = multiprocessing.cpu_count()
    tre_gcv = tree.DecisionTreeClassifier()
    gcv = GridSearchCV(estimator=tre_gcv, param_grid=params, n_jobs=n_cores)
    gcv.fit(X, Y)
    
    tre_gcv = gcv.best_estimator_
    return tre_gcv
        


            
if __name__ == "__main__":
    main()