# CS150A Database Project Code Introduction

## Structure

- Files added are data_cleaning.ipynb, data_visual.ipynb, feature_engineering.py and train.py. The first and the second are for Part 1 and 2 of this project, to explore the data and to check if there is anything to clean. The third file is to do preprocess to both train and test data sets, Part 3 and optional pyspark. The last one is to train the model with the output data set of part 3, this is part 4 and 5 of this project.
- To train the model, you should **run feature_engineering.py first** and check if there is output to ./data/, preprossed_test.csv and preprossed_train.csv. Then you can run train.py to train the model.

## Code Introduction

- I will mainly state the overview of feature_engineering.py and train.py. You can also find code and method explanation in report and also in comments from both files.
- feature_engineering.py
  - We use PySpark to handle the data set operations, especially the user defined functions from pyspark.sql.functions.
  - First remove columns that do not appear in test data set, and divide column 'Problem Hierarchy' into two columns: 'Problem Unit' and 'Problem Section'.
  - Then do naive encoding of specific string columns.
  - Dealing with complex containing ~~. For opportunity(default), take the average of opportunities, then drop opportunity(default). For KC, add the number of KC.
  - Add some new features.  7 new features we have added is Personal Correct First Attempt Count, Personal Correct First Attempt Rate, and Correct First Attempt Rate per Problem, Correct First Attempt Rate per unit, Correct First Attempt Rate per section, Correct First Attempt Rate per step, and also Correct First Attempt Rate per KC. Operations on all these columns resemble.
  - Export the result as preprossed_test.csv and preprossed_train.csv to ./data/.
- train.py
  - Model training. We choose several algorithms such as MLPRegressor, RandomForest and so on, and we evaluate all algorithm by checking the RMSE of them. The optimizing code of RandomForest is also in train() function.
  - Export. Do prediction and export the result to output.csv.
