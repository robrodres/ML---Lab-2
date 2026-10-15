import pandas as pd
import numpy as np
import matplotlib as plt
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

print(f"\n\n\nRows left after removing incomplete rows: {len(data)}\n")


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
print (f"\n\nMissing values in selected features: {data[features].isna().sum().sum()}")