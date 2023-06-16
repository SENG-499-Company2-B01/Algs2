import pandas as pd

path = "data/client_data/"

def excel_to_json(excel_file, json_file):
    data = pd.read_excel(excel_file)
    data.to_json(json_file, orient='records')

    print(f"JSON file '{json_file}' created")

excel_to_json(path+'Course Summary_CSC115.xlsx', path+'Course_Summary_CSC115.json')
excel_to_json(path+'Course Summary_2022_2023.xlsx', path+'Course_Summary_2022_2023.json')
excel_to_json(path+'Course Summary_2019_2021.xlsx', path+'Course_Summary_2019_2021.json')