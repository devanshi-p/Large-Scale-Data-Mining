		EE219 Project-4 REGRESSION ANALYSIS
----------------------------------------------------------------------
				Team members
----------------------------------------------------------------------
Devanshi Patel – 504945601
Ekta Malkan -  504945210
Pratiksha Kap - 704944610
Sneha Shankar – 404946026

This README contains the requirements and steps that are required to execute Project 4.

#########################################
		Dependencies 
#########################################
surprise
csv
pandas
numpy
sklearn
matplotlib
collections

#########################################
		Usage 
#########################################

Each section of the project is done in a separate iPython file inside "Code" folder. The iPython files belonging to this project are as follows:
1) Project4_part1.ipynb : Task 1
2) Project4_LinearRegression.ipynb : Task 2a Linear Regression
3) RFR.ipynb : Task 2b Random Forest
4) Neural Networks.ipynb : Task 2c Neural Networks
5) Ques2-d,e.ipynb : Task 2d and 2e, Workflow wise Linear Regression, Polynomial and KNN Regression

The question numbers have been written as a comment in each cell of the respective ipynb files.

A folder named "plots" must be present in the same directory. This is where the plots get stored for Task 2 a.
(Already a plots folder is present inthis zip with plots for Linear regression).

An Excel file anmed "Results" is also present. It contains Average Train and Average Test RMSE for all 32 combinations for Unregularized, Ridge,Lasso and ElasticNet models for Task 2a.Also, the best hyperparameter values for each combination have been added. It also contains all the intercepts ,coefficients parameter values for 49 features obtained by best combination encoding. 