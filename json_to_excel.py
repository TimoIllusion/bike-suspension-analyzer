import pandas

input_file = "results.json"
output_file = "motobike_analysis_04032023.xlsx"

df = pandas.read_json(input_file)
df.to_excel(output_file)