import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier


# Comment: The objective of this lab is comparing classification models that predict the popularity
# class of a song using its musical characteristics.
# The target variable will have three classes: Low, Medium and High.


# CONTENTS:
#                                               LINE
# 1.- EXPLORING THE DATASET .................... 35
# 2.- DATA PREPARATION.......................... 66
# 3.- CREATE TRAINING AND TEST DATA.............271
# 4.- BUILDING THE CLASIFICATION MODELS.........299
# 5.- TUNING THE MODELS (before comparison).....376
# 6.- COMPARISON AND EVALUATION.................413
# 7.- CONCLUSSION...............................509


# 1.- EXPLORING THE DATASET
# ---------------------------------------------------------------

print("1.- EXPLORING THE DATASET")
print("-"*30)

df = pd.read_csv("comercial.csv")
print(f"\n\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

print("\n\nFirst 20 rows:")
display(df.head(20))

df.info()
display(df.describe(include="all").T)

# Cleaning/filetering data

missing_values = df.isna().sum()

print("\n\n\nMissing values by column: \n")
display(missing_values[missing_values >0])

print(f"\nExact duplicated rows: {df.duplicated().sum()}")

repeated_tracks = df.duplicated(
    subset = ["artist", "song"]
).sum()

print(f"\n\nRepeated artist-song rows: {repeated_tracks}\n")

# 2.- DATA PREPARATION
# ---------------------------------------------------------------

print ("\n\n\n\n2.- DATA PREPARATION")
print ("-"*30)

# Removing rows which Popularity value is not valid
data = df.dropna(subset=["popularity"]).copy()

print(f"\n\n\nRows left after removing incomplete fields: {len(data)}\n")


# Selection of features
features = [
    "total_track_album",
    "danceability",
    "energy",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

print (f"\nSelected features: \n")
for feat in features:
    print(feat)

# Checking popularity/selected musical feat change between observations of the same track

columns_to_check = ["popularity"] + features

inconsistent_tracks = (
    data.groupby(["artist", "song"])[columns_to_check]
    .nunique()
    .gt(1)
    .any(axis=1)
    .sum()
)

print(f"\n\nArtist-song groups with inconsistent target/features: {inconsistent_tracks}")

# Keeping one row per artist-song
data = data.drop_duplicates(
    subset=["artist", "song"]
).copy()

print (f"\n\nRows after keeping unique tracks: {len(data)}")

# Checking missing values in selected feats
print (f"\n\nMissing values in selected features: {data[features].isna().sum().sum()}\n\n")


# Comment: now that all the information is filtered, it is clear that there are no
# incomplete rows left in the dataset.

# Comment: The original Popularity variable is numerical, so it must be converted
# into a categorical variable.
# The three classes, as previously noted, are:
# - Low: 0-49
# - Medium: 50-74
# - High: 75-100

popularity_bins = [0, 50, 75, 101]
class_names = ["Low", "Medium", "High"]

# Conversion
data ["popularity_class"] = pd.cut(
    data ["popularity"],
    bins = popularity_bins,
    labels = class_names,
    right = False,  # this line will includde tracks with score=limit in the highest category
    include_lowest = True
)

# Number of tracks per class
class_counts = (
    data["popularity_class"]
    .value_counts()
    .reindex(class_names)
)
print ("\n\nNumber of tracks per class: ")
display (class_counts.to_frame("Number of tracks"))

# Show proportion % of total tracks per class
class_percentages = (
    data["popularity_class"]
    .value_counts(normalize = True)
    .reindex (class_names) * 100
)
display(class_percentages.round(2).to_frame("Track proportion (%)"))

# Comment: The Medium class is the biggest, containing 43.5% of the tracks in the dataset.



# Checking that most of musical features have homogeneous ranks (0-1)
display (data[["popularity"] + features].describe().T)

bounded_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]


# Searching values out of the range set
print ("\n\nValues outside the expected [0,1] interval: ")

for column in bounded_features:
    outside = ((data[column] <0) | (data[column] >1)).sum()
    print(column, ":", outside)


# Comment: the musical features of all the rows are inside range. Some other features, like tempo
# or key, follow their own score systems, so they have not been included in this filter.



# Representation
# -------------------------


# Distribution of Song Popularity
plt.figure(figsize=(8, 5))

plt.hist(
    data["popularity"],
    bins=15,
    edgecolor="black"
)

plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Tracks")

plt.show()



# Number of Tracks per Popularity Class
plt.figure(figsize=(7, 5))

class_counts.plot(kind="bar")

plt.title("Number of Tracks per Popularity Class")
plt.xlabel("Popularity Class")
plt.ylabel("Number of Tracks")
plt.xticks(rotation=0)

plt.show()


# Relation between popularity and two of the musical features
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, column, label in zip(
    axes,
    ["loudness", "acousticness"],
    ["Loudness", "Acousticness"]
):
    grouped_values = [
        data.loc[
            data["popularity_class"] == class_name,
            column
        ].dropna()
        for class_name in class_names
    ]

    ax.boxplot(grouped_values)
    ax.set_title(f"{label} by Popularity Class")
    ax.set_xlabel("Popularity Class")
    ax.set_ylabel(label)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(class_names)

plt.tight_layout()
plt.show()


# Comment: After cleaning the data taking out repeated observations for the same song and
# rows without popularity score, 600 unique tracks remain.
# The medium class contains the larger amount of tracks. At the same time, all three classes
# are reasonably represented.

# The numerical analysis suggests that "loudness" and "energy" tend to have positive relationships
# with popularity while "acousticness" has a negative relationship, however these relationships are
# not strong enough to guarantee 100% accurate predictions.

# Artists/Songs/Albums titles and names will not be used as input features since they could mistrack
# the models into memorizing identities instead of focusing on relationships among features.







# 3.- CREATE TRAINING AND TEST DATA
# ---------------------------------------------------------------


# input features from before
x = data[features].copy()

# classification target
y = data["popularity_class"].copy()

# split data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.20,   #as in lab 1, 20% for the test, 80% for training
    random_state = 42,
    stratify = y
)


print("\n\n\n3.- CREATE TRAINING AND TEST DATA")
print ("-"*30)
print (f"\n\nTraining set shape: {x_train.shape}")
print (f"\n\nTest set shape: {x_test.shape}\n\n\n")




# 4.- BUILDING THE CLASIFICATION MODELS
# ---------------------------------------------------------------

# Comment: The two classification algorithms used will be:
#   - K-Nearest Neighbors/KNN: classifies a track by looking at adjacent tracks
#       in the training data. Feature scaling is necessary because the numerical
#       variables have different scales.
#   - Decision Tree: can represent nonlinear relationships and doesn't require
#       feature scaling.

#   The same training data will be used for both models.7

print ("4.- BUILDING THE CLASIFICATION MODELS")
print ("-"*30)
print("\n\n")


models = {
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(random_state = 42))
    ])
}

# stratified five fold validation
cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)


# initial cross validation score for each model
baseline_cv_scores = {}

for model_name, model in models.items():

    scores = cross_val_score(   #trains model with different sets
        model,
        x_train,
        y_train,
        cv = cv,
        scoring = "accuracy"
    )

    baseline_cv_scores[model_name] = scores.mean()

    print (f"\nModel: {model_name}\n")
    print (f"Scores for the five folds: {scores.round(3)}\n")
    print (f"Mean F1-macro: {round(scores.mean(), 3)}\n\n")


# hyperparameters

param_grids = {
     "K-Nearest Neighbors": {
        "model__n_neighbors": [3, 5, 7, 9, 11]  # number of neighbor tracks used for classification
    },

    "Decision Tree": {
        "model__max_depth": [3, 5, 10, None],   # limits depth
        "model__min_samples_leaf": [1, 5, 10]   # set leaves
    }
}

# best models and cross validation scores
best_models = {}
tuned_cv_scores = {}
best_parameters = {}




# 5.- TUNING THE MODELS (before comparison)
# ---------------------------------------------------------------



# tuning the models using the training data
for model_name, model in models.items():

    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grids[model_name],
        cv = cv,
        scoring = "accuracy",
        refit = True    #once best config is chosen(cv) the model trains again with all the traning set
    )

    grid_search.fit(x_train, y_train)

    # best version of each model
    best_models[model_name] = grid_search.best_estimator_

    # best cross validation score
    tuned_cv_scores[model_name] = grid_search.best_score_

    # best hyperparameters
    best_parameters[model_name] = grid_search.best_params_

    print("\nModel:", model_name)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:",
          round(grid_search.best_score_, 3))
    print("\n\n")





# 6.- COMPARISON AND EVALUATION
# ---------------------------------------------------------------

comparison = pd.DataFrame ({
    "Model": list (models.keys()),
    "Baseline CV Accuracy": [
        baseline_cv_scores[name]
        for name in models
    ],
    "Tuned CV Accuracy": [
        tuned_cv_scores[name]
        for name in models
    ]
})

comparison["Improvement"] = (
    comparison["Tuned CV Accuracy"] - comparison["Baseline CV Accuracy"])

display (comparison.round(3))

print ("\n\n")


# final results
final_metrics = []

for model_name, model in best_models.items():


    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)


    precision = precision_score(
        y_test,
        y_pred,
        average = "weighted",
        zero_division = 0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average = "weighted",
        zero_division = 0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average = "weighted",
        zero_division = 0
    )

    final_metrics.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

#display
final_results = pd.DataFrame(final_metrics).set_index("Model")

display(final_results.round(3))

print("\n\n")

# condufsion matrix
for model_name, model in best_models.items():

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred, labels = class_names
    )

    print(f"Confusion Matrix - {model_name}")

    display (pd.DataFrame(cm,
                          index = class_names,
                          columns = class_names))

    # visual
    display_matrix = ConfusionMatrixDisplay(confusion_matrix = cm,
                                            display_labels = class_names)

    display_matrix.plot(values_format = "d")

    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()




# 7.- CONCLUSSION
# ---------------------------------------------------------------
#
# Comment: In this lab, K-Nearest Neighbors (KNN) and Decision Tree (DT) were used to classify music tracks
#   into three popularity classes: Low, Medium, High.
#   Both models were evaluated using 5-fold cross-validation and tuned using different hyperparameters.
#
#   Before tuning, KNN achieved a CV accuracy of 37,9%, which increased slightly to 38,1% after tuning.
#   DT improved from 34,3% to 41%. Therefore, DT achieved the highest CV acc. after tuning   
#
#   However the final results were different: KNN achieved an acc. of 39,2% while DT achieved 32,5%.
#   KNN also achieved highest weighetd Precision, Recall and F1 score than DT:
#   
#   Model       Precision   Recall  F1-score
#   KNN         39,4%       39,2%   39,3%
#   DT          32,8%       32,5%   32,5%  
#
#
#   The confusion matrices show that KNN classified the Medium class most correctly, succesfully identifying
#   25 out of 52 tracks. For the Low class, both models correctly classified 11 out of 29. For the High class,
#   both correctly classified as well 11 out of 39.
#   On the other hand, both models present difficulties when identifying high popular songs, where KNN incorrectly
#   classified 18 High tracks as Medium and the DT made the same mistake with 20 tracks.
#
#   Neither model provides a strong result nor improvement for this dataset, but KNN suits it better due to the 
#   better results, eventhough the performance is still limited.
#   Overall, the results suggest that the provided musical features in the dataset would not be enough
#   data to feed a reliable popularity prediction.
