import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display


# Comment: The objective of this lab is comparing classification models that predict the popularity
# class of a song using its musical characteristics.
# The target variable will have three classes: Low, Medium and High.


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