import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt

# Importing the dataset
training = pd.read_csv('Training.csv')
testing  = pd.read_csv('Testing.csv')

cols = training.columns[:-1]

x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

testx = testing[cols]
testy = testing['prognosis']  
testy = le.transform(testy)

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []
    
    print("Healtho: Please describe the symptoms you're experiencing.")
    symptoms_input = input().lower()
    
    for symptom in cols:
        if symptom.lower() in symptoms_input:
            symptoms_present.append(symptom)
    
    present_disease = clf.predict([[(symptom in symptoms_present) for symptom in cols]])[0]
    present_disease = le.inverse_transform([present_disease])[0]
    
    print("Healtho: Based on the symptoms you described, you may have " + present_disease)
    red_cols = reduced_data.columns 
    symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
    print("Healtho: Other symptoms commonly associated with " + present_disease + " include:")
    print(symptoms_given)
    
    # Additional actions based on risk level or recommendation to consult a doctor can be added here

flag=True
print("Healtho: Hello! I'm Healthmate. I'm here to help you with your health concerns. If you want to exit, just say 'Bye'.")
while(flag==True):
    
    user_response=input()
    user_response=user_response.lower()
     
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Healtho: You're welcome!")
       
        else:
            if(greeting(user_response)!=None):
                print("Healtho: "+greeting(user_response))
                tree_to_code(clf, cols)
            else:
                print("Healtho: I'm sorry, I didn't understand. Could you describe your symptoms?")
                
    else:
        flag=False
        print("Healtho: Goodbye! Take care.")
