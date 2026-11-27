import pandas as pd

# Load the updated validation file
updated_file = 'text_with_openAI_Validation_updated.csv'
validation_df = pd.read_csv(updated_file)

# Count the number of rows where 'user_cat' is not 'None'
non_none_count = validation_df[validation_df['user_cat'] != 'None'].shape[0]

print(f"Number of lines in 'user_cat' that aren't 'None': {non_none_count}")
