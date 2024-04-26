import pickle
import pandas as pd

with open('decision_tree_ccp_model.pkl', 'rb') as file:
    decision_tree_model, features_list = pickle.load(file)
    
def predict_gender(noun):
    #print(features_list)
    noun = '<S>' + noun + '<E>'
    features = {col: bool(col in noun) for col in features_list}
    features = pd.DataFrame(features, index=[0])

    #print(features)
    #print(decision_tree_model)
    
    gender = decision_tree_model.predict(features)[0]
    
    gender_mapping = {
    'f': 'die',
    'm': 'der',
    'n': 'das'
    }
    
    return gender_mapping.get(gender)