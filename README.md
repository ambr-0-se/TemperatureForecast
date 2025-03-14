# Evaluation of Graph Neural Network on Temperature Forecast

## Introductioon
Weather forecast has been playing a critical role in contemporary society, and improving the accuracy of temperature forecast could bring numerous social and economic benefits. In this project, our objective is to evaluate different techniques, and these results are going to serve as the foundation for more complicated weather prediction task in the future. 

We retrieved a weather dataset from European Climate Assessment & Dataset (ECA&D), which contains the daily observations from 2000-2010 at 18 meteorological stations in Europe. The objective is to train models that predict the mean temperature of the coming day using previous spectral and temporal features.

We experimented with different machine learning models and graph neural networks to predict the mean temperature. We evaluated the accuracy of different models with mean square error, and examined how number of layers and batch size affect the performance of the model.


![region graph with 1NN](region_graph.png?height=300)

## Explanation of the file structure
1. `/data` stores the original location and the daily observations of the 18 stations 
2. `/feature` stores the graph containing 18 staions with attributes, and graphs (with node and edge features) split into train, dev and test set
3. `/result` stores the new evaluation result of different models and hyperparameter settings
4. `/result_comparison` is a collection of previous results that compare against different models `/result_comparison/initial`, and the impact of number of edges per station, temporal window, number of layers and batch size

## To run the code
1. Download or clone the directory
2. Navigate to the directory `/TemperatureForecast`
3. Set up the conda environment with ```conda env create -f environment.yml``` and activate the environment with ```conda activate weather_forecast```
4. Adjust the hyperparameters in `config.py`
5. Train and evaluate the models with ```python main.py```
6. Check the results in `\results`

## References
[1] Klein Tank, A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment. Int. J. of Climatol., 22, 1441-1453. Data and metadata available at http://www.ecad.eu

Florian Huber, Dafne van Kuppevelt, Peter Steinbach, Colin Sauze, Yang Liu, Berend Weel, "Will the sun shine? – An accessible dataset for teaching machine learning and deep learning", https://proceedings.mlr.press/v207/huber23a/huber23a.pdf
