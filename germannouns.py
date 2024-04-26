import re
import pandas as pd
from pprint import pprint
import csv
import numpy as np
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle


### Open using panda dataframe ###

def main():
    
    nouns = pd.read_csv('nouns.csv')
    
    n_cores = multiprocessing.cpu_count()

    ### Extract names and gender ###

    nouns = nouns[["lemma","genus"]]
    #print(nouns.shape)

    nouns = nouns.dropna()
    
    nouns = nouns.head(2000)

    ### Clean the data to remove anything that starts with numbers and special characters ###

    #print(f"Starting shape is {nouns.shape}")

    for noun in nouns["lemma"]:
        if re.search(r"^[\W]", str(noun)):
            nouns.drop(nouns[(nouns["lemma"] == noun)].index, inplace=True)


    #print(f"Current shape is {nouns.shape}")

    #print(nouns.head(10))
    
    ### Build a list of characters for each word
    nouns['Chs'] = ""
    for noun in nouns["lemma"]:
        chs = ['<S>'] + list(str(noun)) + ['<E>']
        #print(chs)
        i = (nouns.index[nouns['lemma'] == noun].tolist())[0]
        #print(i)
        nouns.at[i, 'Chs'] = chs
    
    
    #print("Character breakdown complete...")
    #print(f"Current shape is {nouns.shape}")
    
    nouns = nouns.dropna()
    
    #print(f"Current shape is {nouns.shape}")
    
    Names = {2: 'Chs2', 3: 'Chs3', 4: 'Chs4', 5: 'Chs5', 6: 'Chs6'}
    
    for i in range(2, 6):
        nouns[Names[i]] = ""
        
    for noun in nouns["lemma"]:
        i = (nouns.index[nouns['lemma'] == noun].tolist())[0]
        for n in range(2, 6):
            tples = create_ngram(noun, n)
            nouns.at[i, Names[n]] = tples
    
    #print(nouns.head(5))
    #print(f"Current shape is {nouns.shape}")
    
    '''
    The next chunk of code creates sets of unique n-tuples
    from the list of n-tuples generated above.  
    
    '''
    
    headers = list(nouns.columns[2:])
    #print("Headers are: ", headers)
    
    sets_chs = []
    
    for count, header in enumerate(headers):
        tempset = set()
        for tpl in nouns[header]:
            #print(type(tpl))
            for l in tuple(tpl):
                #print(l, sep= ", ")
                if l not in tempset:
                    tempset.add(l)
        #print("") 
        #print("Length of tempset:", len(tempset))
        sets_chs.append(tempset.copy()) 
        tempset.clear()
    
    #print(len(sets_chs))
    
    '''
    This chunk of code calculates the number of instances of
    each gender for each entry in the list of tuples in sets_chs. 
    For example, it calculates how often the character 's' may 
    appear in the m, f or neutral gender.
    
    '''
    
    '''
    for noun in nouns["Chs"]:
        print(noun)
        filtered_list = nouns.loc[nouns['Chs'].apply(lambda x: x == noun)]
        print(filtered_list['genus'])
    '''
    
    lst_freq_dict = []
    for count, header in enumerate(headers):
        freq_dict = {}
        for chr in sets_chs[count]:
            for noun in nouns[header]:
                if not noun:
                    continue
                #print(chr, noun)
                if chr not in noun:
                    continue
                #print(chr, noun)
                #print(nouns.loc[nouns['Chs'].apply(lambda x: x == noun)]['genus'].values[0])
                n_gram = (chr, (nouns.loc[nouns[header].apply(lambda x: x == noun)])['genus'].values[0]) 
                #print(n_gram)
                freq_dict[n_gram] = freq_dict.get(n_gram, 0) + 1
        lst_freq_dict.append(freq_dict.copy())
        freq_dict.clear()

    #print(lst_freq_dict)
    
    sorted_dicts = []
    for dictionary in lst_freq_dict:
        #print(dictionary)
        sorted_dict = sorted(dictionary.items(), key = lambda x: x[1], reverse=True) 
        sorted_dicts.append(dict(sorted_dict.copy()))
        sorted_dict.clear()
    
    '''
    This creates a few summary statistics for the entire dataset
    
    '''
    
    categories = ['f', 'm', 'n']
    
    genus_counts = nouns['genus'].value_counts(normalize=True)
    print(genus_counts)
    
    '''
    This bit creates the absolute counts and relative frequency distributions 
    for each n-gram in the dataset (where applicable)
    
    '''
    
    #print(sorted_dicts[0])
    lst_chs_counts = []
    
    for count, st in enumerate(sets_chs):
        chs_counts = {}
        for chs in st:
            chs_counts[chs] = sum((sorted_dicts[count]).get((chs, category), 0) for category in categories)
        lst_chs_counts.append(chs_counts.copy())
        chs_counts.clear()
    
    '''
    
    Create a list of dicts of absolute counts
    
    '''
    
    lst_absolute_counts = []
    
    for count, itm in enumerate(lst_chs_counts):
        absolute_values ={}
        for word, total_count in lst_chs_counts[count].items():
            absolute_values[word] = {category: sorted_dicts[count].get((word, category), 0) for category in categories}
        lst_absolute_counts.append(absolute_values.copy())
        absolute_values.clear()
    
    print(lst_absolute_counts[0])
    
    
    '''
    
    Create a list of dicts of relative counts
    
    '''
    
    
    lst_relative_values = []
    
    for count, itm in enumerate(lst_chs_counts):
        relative_values ={}
        for word, total_count in lst_chs_counts[count].items():
            relative_values[word] = {category: sorted_dicts[count].get((word, category), 0) / total_count for category in categories}
            relative_values[word]['Total'] = total_count
        #print(relative_values)
        lst_relative_values.append(relative_values.copy())
        relative_values.clear()
        
    '''
    
    This converts the mess of nested dicts into dataframes we can now use to test
    
    '''
    
    absolute_counts_df = pd.DataFrame()
    
    
    for count, lst in enumerate(lst_absolute_counts):
        absolute_counts_df = pd.concat([absolute_counts_df, pd.DataFrame.from_dict(lst_absolute_counts[count], orient = 'index')], axis = 0)
        
    absolute_counts_df.reset_index(inplace=True)
    absolute_counts_df.rename(columns={'index': 'n-gram'}, inplace=True)
    
    #print(absolute_counts_df)
    
    relative_values_df = pd.DataFrame()
    
    for count, lst in enumerate(lst_relative_values):
        relative_values_df = pd.concat([relative_values_df, pd.DataFrame.from_dict(lst_relative_values[count], orient = 'index')], axis = 0)
        
    relative_values_df.reset_index(inplace=True)
    relative_values_df.rename(columns={'index': 'n-gram'}, inplace=True)
    
    print(len(relative_values_df))
    
    #print(relative_values_df)
    
    
    sorted_relative_values_f = relative_values_df[relative_values_df['Total'] > 10].sort_values(by = 'f', ascending=False)
    #print("Sorted values by F:")
    #print(sorted_relative_values_f.head(20))
    
    sorted_relative_values_m = relative_values_df[relative_values_df['Total'] > 10].sort_values(by = 'm', ascending=False)
    #print("Sorted values by M:")
    #print(sorted_relative_values_m.head(20))

    sorted_relative_values_n = relative_values_df[relative_values_df['Total'] > 10].sort_values(by = 'n', ascending=False)
    #print("Sorted values by N:")
    #print(sorted_relative_values_n.head(20))
    
    '''
    
    Conduct chi-square test and return p-value for each row
    
    '''

    relative_values_df['P_Value'] = relative_values_df.apply(lambda row: calculate_p_value(row, genus_counts), axis=1)  
    sorted_relative_value_pvalue = relative_values_df[relative_values_df['Total'] > 10].sort_values(by = 'P_Value', ascending=True)
    print(len(sorted_relative_value_pvalue))
    
    '''
    FEATURE REDUCTION
    
    This piece of code looks at the relative frequencies of n-grams and 
    drops those below a certain threshhold    
    
    '''
    
    count_NAN = sorted_relative_value_pvalue['P_Value'].isna().sum()
    proportion_NAN = count_NAN / len(sorted_relative_value_pvalue)
   
    reduced_features = sorted_relative_value_pvalue.dropna()
    
    #print(reduced_features.head(10))
    
    decisiontree_df = pd.DataFrame()
    
    decisiontree_df['Word'] = '<S>' + nouns['lemma'] + '<E>' # Populate list of words
    decisiontree_df['Genus'] = nouns['genus']
    
    columns = reduced_features['n-gram'].apply(lambda x: ''.join(x)) # Extract features from reduced_features list
    
    for col in columns:
        decisiontree_df[col] = '' # Create columns for each feature
        decisiontree_df[col] = decisiontree_df['Word'].apply(lambda x: col in x)
    
        
    #print(decisiontree_df.head(10))
    
    '''
    
    This block of code implements the decision tree. The dataset is split into 
    three parts: training, test and validation.
    
    '''
    
    Y = decisiontree_df['Genus']
    X = decisiontree_df.drop(columns=['Word' , 'Genus'])
    
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, Y, test_size=0.4, random_state=20)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)
    
    tre = DecisionTreeClassifier()
    
    tre.fit(X_train_temp, y_train_temp)
    
    # Initialize a DataFrame to store results
    results = []
    
    path = tre.cost_complexity_pruning_path(X_train_temp, y_train_temp)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    
    tres = Parallel(n_jobs=n_cores)(
    delayed(train_decision_tree)(ccp_alpha, X_train_temp, y_train_temp) for ccp_alpha in ccp_alphas)
    
    '''
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train_temp, y_train_temp)
        tres.append(clf)
    
    '''
    
    # Remove the last element in clfs and ccp_alphas and plot trees
    
    tres = tres[:-1]
    ccp_alphas = ccp_alphas[:-1]
    node_counts = [tre.tree_.node_count for tre in tres]
    depth = [tre.tree_.max_depth for tre in tres]
    plt.scatter(ccp_alphas,node_counts)
    plt.scatter(ccp_alphas,depth)
    plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
    plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
    plt.legend()
    plt.show()
    
    # Trying to figure out which alpha to choose
    
    train_acc = []
    test_acc = []
    for c in tres:
        y_train_pred = c.predict(X_train_temp)
        y_test_pred = c.predict(X_test)
        train_acc.append(accuracy_score(y_train_pred,y_train_temp))
        test_acc.append(accuracy_score(y_test_pred,y_test))

    plt.scatter(ccp_alphas,train_acc)
    plt.scatter(ccp_alphas,test_acc)
    plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
    plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
    plt.legend()
    plt.title('Accuracy vs alpha')
    plt.show()
    
    tre_ = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.0005)
    tre_.fit(X_train_temp, y_train_temp)
    y_train_pred = tre_.predict(X_train_temp)
    y_test_pred = tre_.predict(X_test)
    
    print(f'Train score {accuracy_score(y_train_pred,y_train_temp)}')
    print(f'Test score {accuracy_score(y_test_pred,y_test)}')
    
    '''
    # Visualize the tree
    plt.figure(figsize = (50,50))
    plot_tree(tre_, filled=True, feature_names=X.columns, class_names=Y.unique())
    plt.show()
    
    # Analyze feature importance
    print("Feature Importance:")
    for feature, importance in zip(X_train_temp.columns, tre_.feature_importances_):
        print(f"{feature}: {importance}")

    # Analyze node distribution
    print("\nNode Distribution:")
    n_nodes = tre_.tree_.node_count
    children_left = tre_.tree_.children_left
    children_right = tre_.tree_.children_right
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of leaf nodes: {sum(children_left == -1)}")

    # Analyze tree depth
    depth = tre_.get_depth()
    print(f"Tree depth: {depth}")

    # Analyze leaf node statistics
    leaf_samples = tre_.tree_.n_node_samples[children_left == -1]
    class_distribution = tre_.tree_.value[children_left == -1]
    print("\nLeaf Node Statistics:")
    for i, (samples, distribution) in enumerate(zip(leaf_samples, class_distribution)):
        print(f"Leaf Node {i}:")
        print(f"  Samples: {samples}")
        print(f"  Class Distribution: {distribution}")
    
    '''
    
    '''
    
    Prepruning the tree using GridSearch
    
    '''
    
    params = {'max_depth': [*range(1,10)],'min_samples_split': [2,3,4], 'min_samples_leaf': [1,2]}
    
    tre_gcv = tree.DecisionTreeClassifier()
    gcv = GridSearchCV(estimator=tre_gcv, param_grid=params, n_jobs=n_cores)
    gcv.fit(X_train_temp, y_train_temp)
    
    tre_gcv = gcv.best_estimator_
    tre_gcv.fit(X_train_temp, y_train_temp)
    y_train_pred = tre_gcv.predict(X_train_temp)
    y_test_pred = tre_gcv.predict(X_test)
    
    print(f'Train score {accuracy_score(y_train_pred,y_train_temp)}')
    print(f'Test score {accuracy_score(y_test_pred,y_test)}')
        
    for max_depth in range(1, 3):
        tre = DecisionTreeClassifier(max_depth=max_depth)
        
        tre.fit(X_train_temp, y_train_temp)
        
        y_pred_test = tre.predict(X_test)
        
        accuracy_test = accuracy_score(y_test, y_pred_test)
        
        results.append({'Max Depth': max_depth, 'Accuracy': accuracy_test})
        
    plt.figure(figsize = (20,20))
    plot_tree(tre, filled=True, feature_names=X.columns, class_names=Y.unique())
    plt.show()

    print("Decision Tree Results:")
    print(results)
    
    '''
    
    Random forest classifier
    
    '''
        
    # Initialize Classifier
    rf = RandomForestClassifier(n_estimators=100, n_jobs=n_cores, random_state=42)
    
    # Train the Classifier
    rf.fit(X_train_temp, y_train_temp)
    
    # Predict the target value
    y_test_pred_rf = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_test_pred_rf)
    
    print("Accuracy:", accuracy)
    

   
''' 

Create a function that builds 2-, 3- and 4-grams from the word,
along with the special characters that indicate the start and 
end of the word

'''

def create_ngram(word, n):
    chr = ['<S>'] + list(str(word)) + ['<E>']

    chs = []
    for i in range(0, n):
        
        chs.append(chr[i:])

    chs = list(tuple(zip(*chs)))
    return chs
          
          
'''
Function to conduct chi-square test on a given row and extract p-value

'''  

def calculate_p_value(row, expected_proportions):
    observed = row[['f', 'm', 'n']] * row['Total']
    expected = expected_proportions * row['Total']
    
    # Check if all expected counts are above 5, otherwise return NA
    
    if all(count > 5 for count in expected):
        _, p_value, _, _ = chi2_contingency([observed, expected])
        return p_value
    else:
        return np.nan 
    
# For each alpha we will append our model to a list
def train_decision_tree(ccpaplha, xtrain, ytrain):
    dt_classifier = DecisionTreeClassifier(ccp_alpha=ccpaplha, random_state=42)
    dt_classifier.fit(xtrain, ytrain)
    return dt_classifier
            
if __name__ == "__main__":
    main()