import numpy as np
import pandas as pd
import os.path

def read_xl_from_local_dir():
    current_path = os.path.dirname(__file__)
    xl_folder = os.path.join(current_path, "Client_Data")
    # Read from two locally stored excel sheets while we wait on the db connection
    path_enrollment_19_to_21 = os.path.join(xl_folder, "Course Summary_2019_2021.xlsx")
    path_enrollment_22_to_23 = os.path.join(xl_folder, "Course Summary_2022_2023.xlsx")
    enrollment_19_to_21 = pd.read_excel(path_enrollment_19_to_21)
    enrollment_22_to_23 = pd.read_excel(path_enrollment_22_to_23)
    # Combine relevant columns of the two sheets into one dataframe
    columns = ["Term", "Subj", "Num", "Enrolled"]
    data = pd.concat([enrollment_19_to_21[columns], enrollment_22_to_23[columns]])
    # Add semester attribute to data
    semesters = ["Fall", "Spring", "Summer"]
    semester_conditions = [
        (data["Term"].astype(str).str[4:6] == "09"),
        (data["Term"].astype(str).str[4:6] == "01"),
        (data["Term"].astype(str).str[4:6] == "05")
    ]
    data["Semester"] = np.select(semester_conditions, semesters)
    return(data)

"""
def main():
    read_xl_from_local_dir()

if __name__ == "__main__":
    print(pd.__version__)
    main()
"""