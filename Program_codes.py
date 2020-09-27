Program Code 1.1 Loading data from .txt files to the data table
data = pd.read_csv('features.txt', sep="\t", header=None)
data['target'] = pd.read_csv('target.txt', sep="\t", header=None)

Program Code 4.1 Deleting rows from the table where the target column is more than 6000
Code from [1]: 6.Outliers Treatment 
data_train = data_train.drop(data_train[data_train.target>5599].index)
data_train = data_train.drop(data_train[data_train.target<4000].index)
data_train.reset_index(drop = True, inplace = True)
data_train.describe()

Program Code 5.1 Replacing all zeros in the table with np.nan
data = data.replace(0, np.nan)
missing_columns = data.columns[data.isnull().any()].values
total_columns = np.count_nonzero(data_merged.isna().sum())
data.dropna(axis=1, how='any', inplace=True)

Program Code 6.1 Search for 10 most positively/negatively correlated features with the target
Code from [1]: 9.Bivariate Analysis
data_train = data[0:109]
data_corr = data_train.corr()
temp_df = data_corr['target'].sort_values(ascending = False)[:11]
display(temp_df)
temp_df = data_corr['target'].sort_values(ascending = False)[-10:]
display(temp_df)

Program Code 6.2 Reinitializing the data table with features selected by bivariate analy-sis 
data = data[[7,8,10,32,33,34,49,97,100,119,125,126,127,128,147,182,187,191,192,
198,'target']]

Program Code 7.1 Removal of features having correlation coefficients higher that 0.95 
Code from [1]: 9.Bivariate Analysis
corr = data.corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool))
tri = corr.mask(mask)
to_drop = [c for c in tri.columns if any(tri[c] >= 0.95)]
print("Features to drop: ", to_drop)
data = data.drop(to_drop, axis=1)

Program Code 8.1 Wrapper methods using Ridge as a model and feature renaming using actual column names 
X = data.drop('target', axis=1)
y = data['target']
model = Ridge()
rfe = RFE(model, 5)
fit = rfe.fit(X, y)
print("Features number: %s" % (fit.n_features_))
print("Selected: %s" % (fit.support_))
print("Feature rank: %s" % (fit.ranking_))
data = data[[7,97,147,191,192]]
data = data.rename(columns={7:"NGT", 97:"TD", 147:"RCI",191:"glob",192:"QPpolrz"})
data.head(2)

Program Code 9.1 Initialization of regression models and dataset split to X and y
state = 43
linear = LinearRegression(n_jobs = -1)
lasso = Lasso(random_state = state)
ridge = Ridge(random_state = state)
kr = KernelRidge()
elnt = ElasticNet(random_state = state)
dt = DecisionTreeRegressor(random_state = state)
svm = SVR()
knn = KNeighborsRegressor(n_jobs = -1)
rf =  RandomForestRegressor(n_jobs = -1, random_state = state)
et = ExtraTreesRegressor(n_jobs = -1, random_state = state)
ab = AdaBoostRegressor(random_state = state)
gb = GradientBoostingRegressor(random_state = state)
xgb = XGBRegressor(random_state = state, n_jobs = -1)
lgb = LGBMRegressor(random_state = state, n_jobs = -1)
X = dataset.drop('target', axis=1)
y = dataset['target']

Program Code 10.1 Training models on training data and evaluating models on test data
Code from [1]: 11.Model Building & Evaluation 
def rmse_score(model):
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_predict, y_test))
return rmse
models = [lasso, ridge, kr, elnt, dt, svm, knn, rf, et, ab, gb, xgb, lgb]
rmse_scores = []
for model in models:
rmse_scores.append(rmse_score(model))
rmse_scores_pd = pd.DataFrame(data = rmse_scores, columns = ['RMSE'])
rmse_scores_pd.index=['LSO','RIDGE','KR','ELNT','DT','SVM','KNN','RF','ET','AB',
'GB','XGB','LGB']
rmse_scores_pd = rmse_scores_pd.round(5)
x = rmse_scores_pd.index
y = rmse_scores_pd['RMSE']

Program Code 10.2 grid_search function for training and optimizing models
Code from [1]: 11.Model Building & Evaluation 
def grid_search(model, params):
global best_params, best_score
grid_search = GridSearchCV(estimator = model,param_grid=params,cv = 10,
verbose=1, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_ 
best_score = np.sqrt(-1*(np.round(grid_search.best_score_, 5)))
return best_params, best_score

Program Code 10.3 Optimization of Elastic Net model
Code from [1]: 11.Model Building & Evaluation 
elastic_params={'alpha':[0.0003,0.00035,0.00045,0.0005], 
'l1_ratio': [0.80, 0.85, 0.9, 0.95],'random_state':[state]}
grid_search(elnt, elastic_params)
elastic_best_params, elastic_best_score = best_params, best_score

Program Code 10.4 Optimization of Kernel Ridge model
Code from [1]: 11.Model Building & Evaluation 
ker-nel_params = {'alpha':[0.27, 0.28, 0.29, 0.3],'kernel':['polynomial', 'linear'],'degree':[2, 3],'coef0':[3.5, 4, 4.2]}
grid_search(kr, kernel_params)
kernel_best_params, kernel_best_score = best_params, best_score

Program Code 10.5 Optimization of Ridge model
Code from [1]: 11.Model Building & Evaluation 
ridge_params={'alpha':[9,9.2,9.4,9.5,9.52,9.54,9.56,9.58,9.6,9.62,9.64,9.66,9.68,
9.7,9.8],'random_state':[state]}
grid_search(ridge, ridge_params)
ridge_best_params, ridge_best_score = best_params, best_score

Program Code 10.6 Optimization of Lasso model
Code from [1]: 11.Model Building & Evaluation 
alpha = [0.0001, 0.0002, 0.00025, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034, 
0.00035, 0.00036, 0.00037, 0.00038, 0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 
0.0008,  0.001, 0.002,0.005, 0.007, 0.008, 0.01]
lasso_params = {'alpha': alpha,'random_state':[state]}
grid_search(lasso, lasso_params)
lasso_best_params, lasso_best_score = best_params, best_score

Program Code 10.7 Reinitializing objects with optimal hyperparameters
elastic_net = ElasticNet(**elastic_best_params)
kernel_ridge = KernelRidge(**kernel_best_params)
ridge = Ridge(**ridge_best_params)
lasso = Lasso(**lasso_best_params)

Program Code 11.1 Model training and forecasting on test data
def predict_with_optimized_models(model):
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.around(y_pred)
y_pred = y_pred.astype(int)
return y_pred
elastic_net_predicted = predict_with_optimized_models(elastic_net)
kernel_ridge_predicted = predict_with_optimized_models(kernel_ridge)
ridge_predicted = predict_with_optimized_models(ridge)
lasso_predicted = predict_with_optimized_models(lasso)

Program Code 11.2 Calculating r2 and rmse and comparing predicted values with actual data
from sklearn import metrics
def calc_r2_rmse(model_name, y_predicted):
rmse = np.sqrt(mean_squared_error(y_predicted, y_test))
r2 = metrics.r2_score(y_test,y_predicted)
for model_name, model_output in predicted_values.items():
   calc_r2_rmse(model_name, model_output)
