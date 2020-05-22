# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function is used to extract messages and
    categories data with given paths. Indexes of input should match.
    Input:
        messages_filepath (str): path to csv file of messages
        categories_filepath (str): path to csv file of categories
    Output:
        df (pd.DataFrame): dataframe that includes messages and categories
        
    """
    #extract data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories)
    # creating a dataframe of individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    #renaming columns
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df


def clean_data(df):
    '''This function is use to clean merged messages and 
    categories dataframe. Duplicate values are removed
    Input:
        df (pd.DataFrame): data to be cleaned
    Output:
        df (pd.DataFrame): cleaned dataframe
    '''
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''This function is used to load data to a given
    database.
    Input:
        df (pd.DataFrame): dataframe to be loaded
        database_filename (str): name of the database to build connection
    Output:
        None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesWithCategories', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()