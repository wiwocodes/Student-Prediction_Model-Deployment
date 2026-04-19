import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
st.set_page_config(page_title="Student Predictor", layout="wide")





class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['total_skill_score'] = (
            X['technical_skill_score'] + X['soft_skill_score']
        )

        X['experience_score'] = (
            X['internship_count'] +
            X['live_projects'] +
            X['work_experience_months']
        )

        X['academic_score'] = (
            X['ssc_percentage'] +
            X['hsc_percentage'] +
            X['degree_percentage'] +
            X['cgpa'] * 25
        )

        return X
    
    
clf_model = pickle.load(open("models/best_classification.pkl", "rb"))
reg_model = pickle.load(open("models/best_regression.pkl", "rb"))

st.title("Student Placement & Salary Predictor")

st.markdown("Predict whether a student gets placed and estimate their salary.")


st.sidebar.header("Input Student Data")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])

ssc = st.sidebar.slider("SSC Percentage", 0, 100, 70)
hsc = st.sidebar.slider("HSC Percentage", 0, 100, 70)
degree = st.sidebar.slider("Degree Percentage", 0, 100, 70)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)

exam = st.sidebar.slider("Entrance Exam Score", 0, 100, 70)
tech = st.sidebar.slider("Technical Skill Score", 0, 100, 70)
soft = st.sidebar.slider("Soft Skill Score", 0, 100, 70)

intern = st.sidebar.slider("Internship Count", 0, 10, 1)
projects = st.sidebar.slider("Live Projects", 0, 10, 1)
exp = st.sidebar.slider("Work Experience (Months)", 0, 60, 0)

cert = st.sidebar.slider("Certifications", 0, 10, 1)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)


input_data = pd.DataFrame([{
    'gender': gender,
    'ssc_percentage': ssc,
    'hsc_percentage': hsc,
    'degree_percentage': degree,
    'cgpa': cgpa,
    'entrance_exam_score': exam,
    'technical_skill_score': tech,
    'soft_skill_score': soft,
    'internship_count': intern,
    'live_projects': projects,
    'work_experience_months': exp,
    'certifications': cert,
    'attendance_percentage': attendance,
    'backlogs': backlogs,
    'extracurricular_activities': extracurricular
}])


if st.button("Predict"):

    placement = clf_model.predict(input_data)[0]
    salary = reg_model.predict(input_data)[0]

    st.subheader("Prediction Result")

    # Placement result
    if placement == 1:
        st.success("Student is likely to be placed")
    else:
        st.error("Student is NOT likely to be placed")

    # Probability (ONLY works for some models)
    if hasattr(clf_model, "predict_proba"):
        prob = clf_model.predict_proba(input_data)[0][1]
        st.write(f"Placement Probability: **{prob:.2%}**")

    # Salary display
    st.metric("Estimated Salary (LPA)", f"{salary:.2f}")

    # Optional comparison chart
    st.bar_chart({
        "Estimated Salary": [salary],
        "Benchmark (5 LPA)": [5]
    })