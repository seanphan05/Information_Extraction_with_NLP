# Information Extraction for text data with Natural Language Processing in Python

This project is about applying NLP algorithms to extract information from text data for text mining. The project had been coded with Python and using tools and libraries for NLP and database manipulation such as nltk, csv, sqlite3 ect.

Raw input data include 2 files. MOUs_Compensation is a corpus of 43 MOUs (Memorandum of Understanding) compensation of the City of LA. Another is MOU1_Compensation, an excerpt from one of the contracts (MOU 1) regarding employee compensation. The idea of this project is to extract compensation information of 43 MOUs to create traning and testing sets in order to build up a machine learning model. This model will be able to extract and summarize the bonus information from compensation section. The model then will be apply to MOU1 for infomation extraction. 

Extracted information will then archived in an SQL database and will be called out by users and allow them to look up under certain conditions.

All files including in this project are:
1. Two raw input data in pdf and docx
2. the main Python code
3. Description of tools and libraries used
4. Description of each functions in the code
5. A SQL database of text mining result
6. Additional files with are create during the process 
