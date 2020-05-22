import sys
# import libraries
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

def load_data(database_filepath):
    '''This function is used to extract data from sqlite database.
    Input:
        database_filepath (str): path to database
    Output:
        X (pd.DataFrame): DataFrame that contains messages data
        Y (pd.DataFrame): DataFrame that contains category columns
        category_names (object): category names in data
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM MessagesWithCategories', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''this function is used to tokenize a string to its
    meaningful words.
    Input:
        text (str): text to be tokenized
    Output:
        clean_tokens: tokenized text, list of words
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''This function generates and return a model to predict categories
    of messages.
    Input:
        None
    Output:
        model: GridSearchCV model
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('moc', MultiOutputClassifier(AdaBoostClassifier()))])
    
    parameters = {'moc__estimator__n_estimators': [50,60],
             'vect__max_df': [0.75, 1.0]
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''This function is used to evaluate a multi-output prediction
    by classification_report method. Each column is printed seperately.
    Input:
        model: model to be evaluated
        X_test: values to be used to predict desired categories
        Y_test: actual values
        category_names: name of each output category
    Output:
        None
    '''
    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    
    for i in category_names:
        print("For column " + i + "\n" + classification_report(Y_test[i], y_pred[i], labels=labels))
    
    


def save_model(model, model_filepath):
    '''This function is used to save a model as pkl file.
    Input:
        model: built model to be saved
        model_filepath: pkl file path
    Output:
        None
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()