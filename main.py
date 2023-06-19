import pandas as pd
from IPython.display import display
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import pickle
from tabpfn import TabPFNClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Set the display width to fit the entire table

dataframe = pd.read_csv('path_to_dataset')
df = dataframe.copy()
# display(df)

df.rename(columns={"Age": "age", "Sex": "sex", "ChestPainType": "cp", "RestingBP": "trestbps", "Cholesterol": "chol",
                   "FastingBS": "fbs", "RestingECG": "restecg", "MaxHR": "thalach", "ExerciseAngina": "exang",
                   "Oldpeak": "oldpeak", "ST_Slope": "slope", "HeartDisease": "condition"}, inplace=True)

df['slope'] = df['slope'].map({"Up": 0, "Flat": 1, "Down": 2})
df['exang'] = df['exang'].map({"N": 0, "Y": 1})
df['restecg'] = df['restecg'].map({"Normal": 0, "ST": 1, "LVH": 2})
df['cp'] = df['cp'].map({"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3})
df['sex'] = df['sex'].map({"F": 0, "M": 1})

# print(df.columns)
df_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'condition']

# display(test_data)

df_copy = df.copy()

combined_num = df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'fbs', 'exang']]
combined_cat = df[['cp', 'restecg', 'slope']]

col_cat_comb = ['cp_typicalangina', 'cp_atypical_angina', 'cp_non_anginalpain', 'cp_asymptomatic', 'restecg_normal',
                'restecg_abnormal', 'restecg_hypertrophy',
                'slope_up', 'slope_flat', 'slope_down']

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc_df_comb = pd.DataFrame(encoder.fit_transform(combined_cat), columns=col_cat_comb)

# display(enc_df_comb)

df_clean = combined_num.join(enc_df_comb)
# Fit encoder on the training data and test data
# display(df_copy)
# leave 10 examples for input data
test_data = df_clean.head(10)
df_clean.drop(test_data.index, inplace=True)
df_copy.drop(test_data.index, inplace=True)
test_data.to_csv('test.csv', index=False)
display(test_data)
# Scaler Object
scaler = MinMaxScaler()
# Fit on the numeric training data


df_clean[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(
    df_clean[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

df_clean = df_clean[df_clean.chol != 0]
df_copy = df_copy[df_copy.chol != 0]

# display(df_clean.describe())
display(df_clean.columns)

# Define random state
rs = 42

X = df_clean
y = df_copy['condition'].astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=rs)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ROS sampling
ros = RandomOverSampler(random_state=rs)

X_ros, y_ros = ros.fit_resample(X_train, y_train)
# Visualize classes


# Visualize classes
y_ros.value_counts().plot.bar(color=['green', 'red'])

plt.title('Value Counts Bar Plot')
plt.xlabel('Values')
plt.ylabel('Counts')
plt.grid(True)

# plt.show()


model = TabPFNClassifier(device='cpu', seed=rs)

model.fit(X_ros, y_ros)

print(model.score(X_train, y_train))
y_eval = model.predict(X_test)

prediction_test = accuracy_score(y_test, y_eval)
print('Accuracy', prediction_test)

with open("models/modelTab.pkl", 'wb') as file:
    pickle.dump(model, file)

with open('models/modelTab.pkl', 'rb') as file:
    model = pickle.load(file)

print(model.predict(X_test))
