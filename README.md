## Ashrae Energy Consumption Kaggle Competition

In this competition the goal is to calculate the energy a building uses on a given time and weather data. In our approuch we are using a LSTM to predict the energy consumption based on the data of the time series for the buildings.

todo



### how to use this project:

1. you need to get the data of the of the competition from kaggle.com

2. place the files into the root directory of this project

3. execute 'create_usage_translator.py',  this creates the csv_file 'primary_usage_translator.csv' we need to translate the given primary_use of a building from a string to an integer

4. execute 'prepare_data.py' to merge all the data into the 'prepared_train.csv' file

   todo