                                                     ____         ___      ___
                                                    |    |       |   \    /   |
                                                    |    |       |    \  /    |
                                                    |    |       |     \/     |
                                                    |    |       |  |\    /|  |
                                                    |    |_____  |  | \  / |  |
                                                    |__________| |__|  \/  |__|

READ ME
----------
Welcome to the Find Your Price app! This application takes used car data(found on kaggle.com), and uses it to predict the value of your used car. You will input several characteristics of your vehicle, and using Decision Tree Regression, the app will evaluate the asking price of your car.

AUTHORS
---------
The author of this application is Lili Meddahi, created from inspiration and instruction from Dr. Jeff Kramer. Created in December 2024.

MODELBUILDER.PY
-----------------
This script is the algorithm for evaluating the price of your used car. It reads in the dataset from kaggle, and then cleans and scrubs the data. Afterwards, it pickles the data so it can be used to predict the price of the car in question. Finally, it uses the dataset to generate a prediction model based on the Decision Tree Regression algorithm. It also calculates the mean absolute error of the prediction.

APP.PY
--------
This script connects the back-end(the predictive model) to the front end(UI). This program runs on flask, and the app.py reads the pickled data in, and then uses that to predict the price of your car, based on the data you gave it.

DEPENDENCIES
----------------
- python 3.x
    https://www.python.org/downloads/
- requests module 
    pip install requests
- flask module
    pip install flask


