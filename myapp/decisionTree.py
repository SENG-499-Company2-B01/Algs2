import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_data(file_path):
    df = pd.read_json(file_path)
    df = df[df['Subj'].isin(['SENG', 'CSC', 'ECE'])]
    df = df[df['Sched Type'].isin(['LEC'])]
    df['Year'] = df['Start Date'].str.split('-').str[0].astype(int)
    df['Term'] = df['Term'].astype(str).str.split('.').str[0].str[-2:]
    df['Course'] = df['Subj']+df["Num"]

    # Remove rows with less than 10 enrolled
    df = df[df['Enrolled'] >= 10]

    return df

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

def train_model(X_train, y_train, preprocessing):

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("DecisionTreeRegressor", tree.DecisionTreeRegressor())
    ])

    # Fit the model
    model.fit(X_train, y_train)

    return model

def predict(model, X):
     
    predictions = model.predict(X)

    return predictions

def perform_decision_tree():
    file_path_19_21 = 'Algs2\myapp\modules\Client_Data\Course_Summary_2019_2021.json'
    file_path_22_23 = 'Algs2\myapp\modules\Client_Data\Course_Summary_2022_2023.json'
    df = pd.concat([get_data(file_path_19_21), get_data(file_path_22_23)])

    cat_features = ['Term', 'Subj', 'Course', 'Section', 'Instructor']#, 'Sched Type']
    num_features = ['Year']#, 'Cap']
    target = ['Enrolled']

    # for cat in cat_features:
    #     print(cat + ":")
    #     print(df[cat].value_counts())
    #     # Remove all categories with less than 10 entries
    #     df = df[df.groupby(cat)[cat].transform('count').ge(10)]
    #     print(df[cat].value_counts())
    #     print()

    preprocessing = create_pipeline(cat_features, num_features)

    # X = preprocessing.fit_transform(df.copy()[cat_features + num_features])
    # # Extract features and target variables
    # transform_columns = preprocessing.get_feature_names_out()
    # X = pd.DataFrame(X.toarray(), columns=transform_columns)
    X = df[cat_features + num_features]
    y = df[target]

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, preprocessing)

    # Calculate R-squared score on the validation data
    score = model.score(X_valid, y_valid)
    predictions = predict(model, X_valid)
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
    score = perform_decision_tree()
    print("R-squared score: {:.2f}".format(score))

main()