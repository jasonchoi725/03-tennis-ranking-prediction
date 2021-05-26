import sys
import sklearn
import numpy as np
import os

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")



import pandas as pd

tennis = pd.read_csv("/Users/junsangwon/Desktop/atp_matches_winner.csv")
tennis


tennis_NoNan = (tennis.dropna(thresh=12))
tennis_NoNan


true_copy_tennis_NoNan = tennis_NoNan.copy()


true_copy_tennis_NoNan.isnull().sum()


true_copy_tennis_NoNan.describe()


%matplotlib inline
import matplotlib.pyplot as plt
true_copy_tennis_NoNan.hist(bins=100, figsize=(80, 60))
save_fig("attribute_histogram_plots")
plt.show()


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(true_copy_tennis_NoNan, test_size=0.2, random_state=42)
test_set.head()


true_copy_tennis_NoNan["w_ace"].hist()
train_set["w_ace"].hist()
test_set["w_ace"].hist()



true_copy_tennis_NoNan["ace_cat"] = pd.cut(true_copy_tennis_NoNan["w_ace"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4,5])
true_copy_tennis_NoNan["ace_cat"].value_counts()


true_copy_tennis_NoNan["ace_cat"].hist()


corr_matrix = true_copy_tennis_NoNan.corr()
corr_matrix["winner_rank"].sort_values(ascending=False)


from pandas.plotting import scatter_matrix

attributes = ["winner_rank", "w_df", "w_bpFaced",
              "w_bpSaved", "w_svpt", "w_1stIn", "w_2ndWon", "w_SvGms", "w_1stWon", "minutes", "w_ace", "winner_age" ]
scatter_matrix(true_copy_tennis_NoNan[attributes], figsize=(50, 30))
save_fig("scatter_matrix_plot")


true_copy_tennis_NoNan = train_set.drop("winner_rank", axis=1)
true_copy_tennis_NoNan_labels = train_set["winner_rank"].copy()
true_copy_tennis_NoNan_labels


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])
true_copy_tennis_NoNan_tr = num_pipeline.fit_transform(true_copy_tennis_NoNan)
true_copy_tennis_NoNan_tr


from sklearn.compose import ColumnTransformer

num_attribs = list(true_copy_tennis_NoNan)
cat_attribs = ["w_ace"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs)
    ])
tennis_prepared = full_pipeline.fit_transform(true_copy_tennis_NoNan)
print(tennis_prepared)
print(tennis_prepared.shape)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)


some_data = true_copy_tennis_NoNan.iloc[:5]
some_labels = true_copy_tennis_NoNan_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측:", lin_reg.predict(some_data_prepared))
print("레이블:", list(some_labels))


from sklearn.metrics import mean_squared_error

tennis_predictions = lin_reg.predict(tennis_prepared)
lin_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

true_copy_tennis_NoNan_labels.hist()


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)


tennis_predictions = tree_reg.predict(tennis_prepared)
tree_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(tennis_prepared, true_copy_tennis_NoNan_labels)


tennis_predictions = forest_reg.predict(tennis_prepared)
forest_mse = mean_squared_error(true_copy_tennis_NoNan_labels, tennis_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("scores:", scores)
    print("average:", scores.mean())
    print("standard deviation:", scores.std())

    
lin_scores = cross_val_score(lin_reg, tennis_prepared, true_copy_tennis_NoNan_labels,
                            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


tree_scores = cross_val_score(tree_reg, tennis_prepared, true_copy_tennis_NoNan_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


forest_scores = cross_val_score(forest_reg, tennis_prepared, true_copy_tennis_NoNan_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(tennis_prepared, true_copy_tennis_NoNan_labels)


print(grid_search.best_params_)
print(grid_search.best_estimator_)


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
