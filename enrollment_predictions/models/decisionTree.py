import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json

X = pd.DataFrame()
y = pd.DataFrame()
X_train = pd.DataFrame()
X_valid = pd.DataFrame()
y_train = pd.DataFrame()
y_valid = pd.DataFrame()
data = pd.DataFrame()

cat_features = []#['Term', 'Subj', 'Course', 'Section', 'Instructor']#, 'Sched Type']
num_features = []#['Year']#, 'Cap']
target = []#['Enrolled']

model = None

def add_data(new_data):
    global data
    data = pd.concat([data, new_data])

def get_data(file_path):
    df = pd.read_json(file_path)
    df = df[df['Subj'].isin(['SENG', 'CSC', 'ECE'])]
    df = df[df['Sched Type'].isin(['LEC'])]
    df['Year'] = df['Start Date'].str.split('-').str[0].astype(int)
    df['Term'] = df['Term'].astype(str).str.split('.').str[0].str[-2:]
    df['Course'] = df['Subj']+df["Num"]
    # Add all courses to terms which do not have them for each year and set their enrolled to 0 and instructor to None
    # for year in df['Year'].unique():
    #     for term in df['Term'].unique():
    #         for course in df['Course'].unique():
    #             if len(df[(df['Year'] == year) & (df['Term'] == term) & (df['Course'] == course)]) == 0:
    #                 df = df.append({'Year': year, 'Term': term, 'Course': course, 'Enrolled': 0, 'Instructor': None}, ignore_index=True)
    # df['Prev Enrolled'] = df.groupby(['Course', 'Term'])['Enrolled'].shift(1)

    # Remove rows with less than 10 enrolled
    # df = df[df['Enrolled'] >= 10]

    return df

# NEED TO CHANGE SO IT CALLS BACKEND TO GET DATA
def import_data():
    file_path_19_21 = 'Algs2\\data\\client_data\\Course_Summary_2019_2021.json'
    file_path_22_23 = 'Algs2\\data\\Client_Data\\Course_Summary_2022_2023.json'
    add_data(get_data(file_path_19_21))
    add_data(get_data(file_path_22_23))

def get_training_data():
    global X, y, X_train, X_valid, y_train, y_valid, data

    # Import the data if data is empty
    if data.empty:
        import_data()

    if len(cat_features) == 0 or len(num_features) == 0 or len(target) == 0:
        import_features()

    X = data[cat_features + num_features]
    y = data[target]

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

def change_features(type, features):
    global cat_features, num_features, target

    if type == 'cat':
        cat_features = features
        # Save features to json file
        with open('Algs2\\data\\model_data\\cat_features.json', 'w') as f:
            json.dump(cat_features, f)
    elif type == 'num':
        num_features = features
        # Save features to json file
        with open('Algs2\\data\\model_data\\num_features.json', 'w') as f:
            json.dump(num_features, f)
    elif type == 'target':
        target = features
        # Save features to json file
        with open('Algs2\\data\\model_data\\target.json', 'w') as f:
            json.dump(target, f)

def import_features():
    global cat_features, num_features, target

    # Load features from json file
    with open('Algs2\\data\\model_data\\cat_features.json') as f:
        cat_features = json.load(f)
    with open('Algs2\\data\\model_data\\num_features.json') as f:
        num_features = json.load(f)
    with open('Algs2\\data\\model_data\\target.json') as f:
        target = json.load(f)

def save_all_features():
    # Save features to json file
    with open('Algs2\\data\\model_data\\cat_features.json', 'w') as f:
        json.dump(cat_features, f)
    with open('Algs2\\data\\model_data\\num_features.json', 'w') as f:
        json.dump(num_features, f)
    with open('Algs2\\data\\model_data\\target.json', 'w') as f:
        json.dump(target, f)

def create_pipeline(cat_features, num_features):
    
        cat_pipeline = Pipeline([
            ("Imputer", SimpleImputer(strategy="most_frequent")),
            ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
        ])
    
        num_pipeline = Pipeline([
            ("Imputer", SimpleImputer(strategy="median")),
            ("Scaler", StandardScaler())
        ])
    
        preprocessing = ColumnTransformer([
            ("Cat", cat_pipeline, cat_features),
            ("Num", num_pipeline, num_features)
        ], remainder="passthrough")
    
        return preprocessing

def train_model():

    global model

    get_training_data()

    if len(cat_features) == 0 or len(num_features) == 0 or len(target) == 0:
        import_features()

    model = Pipeline([
        ("preprocessing", create_pipeline(cat_features, num_features)),
        ("DecisionTreeRegressor", tree.DecisionTreeRegressor())
    ])

    # Fit the model
    model.fit(X_train, y_train)

    # Save the model to Model_Data folder
    joblib.dump(model, "Algs2\\data\\model_data\\\\model.pkl")

def import_model():
    global model

    model = joblib.load("Algs2\\data\\model_data\\model.pkl")

def predict(X):

    if model == None:
        import_model()
     
    predictions = model.predict(X)

    return predictions

def perform_decision_tree():
    file_path_19_21 = 'Algs2\\data\\client_data\\Course_Summary_2019_2021.json'
    file_path_22_23 = 'Algs2\\data\\client_data\\Course_Summary_2022_2023.json'
    add_data(file_path_19_21)
    add_data(file_path_22_23)

    # for cat in cat_features:
    #     print(cat + ":")
    #     print(df[cat].value_counts())
    #     # Remove all categories with less than 10 entries
    #     df = df[df.groupby(cat)[cat].transform('count').ge(10)]
    #     print(df[cat].value_counts())
    #     print()

    # preprocessing = create_pipeline(cat_features, num_features)

    # # X = preprocessing.fit_transform(df.copy()[cat_features + num_features])
    # # # Extract features and target variables
    # # transform_columns = preprocessing.get_feature_names_out()
    # # X = pd.DataFrame(X.toarray(), columns=transform_columns)
    # X = data[cat_features + num_features]
    # y = data[target]

    # # Split the data into training and validation sets
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



    train_model(data)

    # Calculate R-squared score on the validation data
    score = model.score(X_valid, y_valid)
    predictions = predict(X_valid)
    for i in range(len(predictions)):
        # Print the Course, Year and Term 
        print(X_valid.loc[X_valid.index[i], 'Course'], X_valid.loc[X_valid.index[i], 'Year'], X_valid.loc[X_valid.index[i], 'Term'])
        print("Predicted: {:.2f} Actual: {:.2f}".format(predictions[i], y_valid.iloc[i, 0]))
        print()

    # Calculate accuracy and precision and recall and print them
    # accuracy = accuracy_score(y_valid, predictions)
    # precision = precision_score(y_valid, predictions, average='weighted')
    # recall = recall_score(y_valid, predictions, average='weighted')
    # print("Accuracy: {:.2f} Precision: {:.2f} Recall: {:.2f}".format(accuracy, precision, recall))

    return score

def main():
    # train_model()
    get_training_data()
    predictions = predict(X_valid)
    for i in range(len(predictions)):
        # Print the Course, Year and Term 
        print(X_valid.loc[X_valid.index[i], 'Course'], X_valid.loc[X_valid.index[i], 'Year'], X_valid.loc[X_valid.index[i], 'Term'])
        print("Predicted: {:.2f} Actual: {:.2f}".format(predictions[i], y_valid.iloc[i, 0]))
        print()
    # score = perform_decision_tree()
    # print("R-squared score: {:.2f}".format(score))

main()