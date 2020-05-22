# Disaster Response Pipeline Project
### Motivation
This project aims to make some data engineering work and contains extract-transform-load steps and machine learning pipelines. 
1. **ETL Pipelines:** Data is extracted from given csv files. Then they are transformed to be trained in a machine learning model. At last, tranformed data is stored in sqllite database. ETL steps are realized in 'data' folder.
1. **Machine Learning Pipelines:** This pipeline loads data from the database built in ETL steps. Then a multi-output model is generated. Then the data is trained with this et-itf model and the results are evaluated with classification  report. At last, model is stored in a pkl file to be used in front-end.
### Install
Python 3 programming language. All necessary libraries are installed within the program.
* Udacity.com provides needed space for web. Additional setup will be needed to access different spaces.
* 'punkt' and 'wordnet' packages of nltk
### File Descriptions
* **app/templates/ :** This folder contains html document for web interface, which are 'go.html' and 'master.html
* **app/run.py :** Python file that builds back-end of web and execute the program. Ip settings are should be modified here.
* **data/ :** This folder contains the data, ETL processes and loaded sqllite data.
    * **process_data.py :** This file executes ETL steps.
    * **disaster_messages.csv :** This file contains messages data. Both original and translated messages are provided.
    * **disaster_categories.csv :** This file contains categories of messages.
    * **DisasterResponse.db :** Sqllite db that contains 'MessagesWithCategories' table transformed from csv files.
* **models/ :** This folder contains machine learning processings.
    * **train_classifier.py :** Loads data from 'MessagesWithCategories' database, build a proper model trained with the data and save it.
    * **classifier.pkl :** This file stores the model that is built in python file.
 * **licence.txt :** licence file
 * **README.md :** The information file you are currently reading.
 ### Instructions:
1. Run the following comment to obtain spacedomain and space id.
    'env|grep WORK'
1. Above code returns space id and space domain. Replace them in below url. Save it.
    'https://SPACEID-3001.SPACEDOMAIN'
1. Run the following command in the app's directory to run your web app.
    `python run.py`
1. Go to the url found in second matter.
### Licences and Acknowledgements
Thanks to Udacity for Data Science course project. Furthermore, thanks to providers of this open to data to give the opportunity explore new areas.

Please warn me about my mistakes or do not hesitate to give inspiring ideas.

