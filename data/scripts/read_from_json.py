import numpy as np
import pandas as pd
import os.path

def read_cleaned_json(data_folder):
    in_path = os.path.join(data_folder, "Course_Summary_Aggregated_2019_to_2023.json")
    data = pd.read_json(in_path)
    return(data)

def output_cleaned_json(data_folder, data):
    in_path = os.path.join(data_folder, "Course_Summary_Aggregated_2019_to_2023.json")
    data.to_json(out_path)

def read_raw_json_from_local_dir(data_folder):
    # Read from two locally stored json files
    path_enrollment_19_to_21 = os.path.join(data_folder, "Course_Summary_2019_2021.json")
    path_enrollment_22_to_23 = os.path.join(data_folder, "Course_Summary_2022_2023.json")
    enrollment_19_to_21 = pd.read_json(path_enrollment_19_to_21)
    enrollment_22_to_23 = pd.read_json(path_enrollment_22_to_23)
    # Combine relevant columns of the two sheets into one dataframe
    columns = ["Term", "Subj", "Num", "Section", "Enrolled"]
    data = pd.concat([enrollment_19_to_21[columns], enrollment_22_to_23[columns]])
    # Concat subject and course number to new column
    data["CourseName"] = data["Subj"].astype(str) + data["Num"].astype(str)
    # Add semester attribute to data
    semesters = ["Fall", "Spring", "Summer"]
    semester_conditions = [
        (data["Term"].astype(str).str[4:6] == "09"),
        (data["Term"].astype(str).str[4:6] == "01"),
        (data["Term"].astype(str).str[4:6] == "05")
    ]
    data["Semester"] = np.select(semester_conditions, semesters)
    return(data)

def clean_raw_data(raw_data):
    # Remove all sections except unique A0Xs
    raw_data = raw_data.loc[raw_data['Section'].astype(str).str[0] == "A"]
    cleaned_data = raw_data.drop_duplicates(
        subset = ['Term', 'CourseName', 'Section'], 
        keep = 'first'
    ).reset_index(drop = True)
    return(cleaned_data)

def filter_by_seng_program_courses(cleaned_data):
    # Only pull SENG schedule courses from dataset 
    seng_courses = [
        "CSC111","CSC115","ENGR110","ENGR120","ENGR130","ENGR141","MATH100",
        "MATH109","MATH101","MATH110","PHYS111","PHYS110","ECE255","CSC230",
        "CSC225","CHEM101","ECON180","ECE260","ECE310","MATH122","SENG275",
        "SENG265","SENG310","STAT260","ECE458","CSC361","ECE355","CSC355",
        "CSC226","CSC320","ECE360","CSC360","SENG321","CSC370","SENG371",
        "SENG350","SENG360","SENG426","ECE455","CSC460","SENG440","SENG401","SENG499"
    ]
    seng_data = cleaned_data.loc[cleaned_data['CourseName'].isin(seng_courses)]
    return(seng_data)

def sum_course_enrollments(cleaned_data):
    # Sum enrollments for each course
    agg_functions = {'Semester': 'first','Enrolled': 'sum'}
    aggregated_data = cleaned_data.groupby(['Term', 'CourseName']).aggregate(agg_functions)
    aggregated_data = aggregated_data.reset_index()
    return(aggregated_data)

def main():
    # Set up working directory to read and write json files
    path = os.getcwd()
    parent = os.path.dirname(path)
    data_folder = os.path.join(parent, "myapp/modules/Client_Data")
    # Read and clean raw json data
    data = read_raw_json_from_local_dir(data_folder)
    cleaned_data = clean_raw_data(data)
    seng_data = filter_by_seng_program_courses(cleaned_data)
    seng_data_aggregated = sum_course_enrollments(seng_data)
    # Output cleaned data as json file in Client_Data dir
    out_path = os.path.join(data_folder, "Course_Summary_Aggregated_2019_to_2023.json")
    seng_data_aggregated.to_json(out_path)
    # Read cleaned data from new file
    data = read_cleaned_json(data_folder)
    print(data)

if __name__ == "__main__":
    main()
