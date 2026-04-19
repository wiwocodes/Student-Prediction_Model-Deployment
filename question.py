import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def load_data(path):
    df = pd.read_csv(path)
    return df

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        #Skill
        X['total_skill_score'] = (
            X['technical_skill_score'] + X['soft_skill_score']
        )

        # Experience
        X['experience_score'] = (
            X['internship_count'] +
            X['live_projects'] +
            X['work_experience_months']
        )

        # Academic strength
        X['academic_score'] = (
            X['ssc_percentage'] +
            X['hsc_percentage'] +
            X['degree_percentage'] +
            X['cgpa'] * 25
        )

        return X


def build_preprocessor():
    numeric_features = [
        'ssc_percentage','hsc_percentage','degree_percentage','cgpa',
        'entrance_exam_score','technical_skill_score','soft_skill_score',
        'internship_count','live_projects','work_experience_months',
        'certifications','attendance_percentage','backlogs',
        'total_skill_score','experience_score','academic_score'
    ]
    categorical_features = ['gender', 'extracurricular_activities']
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor

def train_classification(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        with mlflow.start_run(run_name=f"classification_{name}", nested=True):

            pipeline = Pipeline([
                ('feature_engineering', FeatureEngineering()),
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            acc = accuracy_score(y_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)

            print(f"{name} Accuracy:", acc)

            if acc > best_score:
                best_score = acc
                best_model = pipeline

    return best_model, best_score

def train_regression(X_train, X_test, y_train, y_test, preprocessor):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR()
    }

    best_model = None
    best_score = -np.inf

    for name, model in models.items():

        with mlflow.start_run(run_name=f"regression_{name}", nested=True):

            pipeline = Pipeline([
                ('feature_engineering', FeatureEngineering()),
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            print(f"{name} RMSE:", rmse, "R2:", r2)

            if r2 > best_score:
                best_score = r2
                best_model = pipeline

    return best_model, best_score

def main():
    mlflow.set_experiment("Student_Placement_Pipeline")
    with mlflow.start_run(run_name="main_run"):

        df = load_data("B.csv")
        X = df.drop(['student_id','placement_status','salary_package_lpa'], axis=1)

        y_class = df['placement_status']
        y_reg = df['salary_package_lpa']

        X_train, X_test, y_train_c, y_test_c = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        _, _, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        preprocessor = build_preprocessor()
        
        best_clf, clf_score = train_classification(
            X_train, X_test, y_train_c, y_test_c, preprocessor
        )

        best_reg, reg_score = train_regression(
            X_train, X_test, y_train_r, y_test_r, preprocessor
        )

        os.makedirs("models", exist_ok=True)

        with open("models/best_classification.pkl", "wb") as f:
            pickle.dump(best_clf, f)

        with open("models/best_regression.pkl", "wb") as f:
            pickle.dump(best_reg, f)

        mlflow.sklearn.log_model(best_clf, "best_classification_model")
        mlflow.sklearn.log_model(best_reg, "best_regression_model")

        mlflow.log_metric("best_classification_accuracy", clf_score)
        mlflow.log_metric("best_regression_r2", reg_score)


if __name__ == "__main__":
    main()