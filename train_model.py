<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv("players_dataset.csv")

# Use fixed encoding to match utils/encoder.py
role_mapping = {"Batsman": 0, "Bowler": 1, "Wicketkeeper": 2, "All-Rounder": 3}
format_mapping = {"T20": 0, "ODI": 1, "Test": 2}

df["role"] = df["role"].map(role_mapping)
df["format"] = df["format"].map(format_mapping)


X = df.drop(["player_name","selected"],axis=1)
y = df["selected"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train,y_train)

os.makedirs("model",exist_ok=True)

joblib.dump(model,"model/selection_model.pkl")

=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv("players_dataset.csv")

# Use fixed encoding to match utils/encoder.py
role_mapping = {"Batsman": 0, "Bowler": 1, "Wicketkeeper": 2, "All-Rounder": 3}
format_mapping = {"T20": 0, "ODI": 1, "Test": 2}

df["role"] = df["role"].map(role_mapping)
df["format"] = df["format"].map(format_mapping)


X = df.drop(["player_name","selected"],axis=1)
y = df["selected"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train,y_train)

os.makedirs("model",exist_ok=True)

joblib.dump(model,"model/selection_model.pkl")

>>>>>>> ef74a501158176de30af4d53b825b5f87f7f10c4
print("Model trained successfully")