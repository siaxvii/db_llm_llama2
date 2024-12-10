import pandas as pd
import random
import os

def introduce_errors(df, num_errors):
    dirty_df = df.copy()
    row_count = len(df)

    error_log = []

    for _ in range(num_errors):
        row_idx = random.randint(0, row_count - 1)
        col = random.choice(df.columns)

        original_value = dirty_df.loc[row_idx, col]
        error_type = None

        if col in ["id", "number_of_seasons", "number_of_episodes", "vote_count"]:
            #numerical fields: uses placeholder error values (instead of NaN)
            new_value = random.choice([-1, 9999])
            dirty_df.loc[row_idx, col] = new_value
            error_type = "Error Type 1: Numerical field error"
        elif col in ["name", "overview", "genres", "networks"]:
            #text fields: introduces typos
            value = str(dirty_df.loc[row_idx, col]) if pd.notna(dirty_df.loc[row_idx, col]) else ""
            if len(value) > 3:
                pos = random.randint(0, len(value) - 1)
                new_value = value[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + value[pos+1:]
                dirty_df.loc[row_idx, col] = new_value
                error_type = "Error Type 2: Typo in text field"
            else:
                new_value = value
                error_type = "Error Type 2: Typo in text field (no change)"


        elif col in ["vote_average", "popularity", "watchlisted_rating"]:
            #decimal fields: introduces invalid formats
            new_value = random.choice([None, -5.0])
            dirty_df.loc[row_idx, col] = new_value
            error_type = "Error Type 3: Invalid decimal format"


        elif col in ["first_air_date", "last_air_date"]:
            #date fields: introduces invalid dates
            new_value = random.choice(["32/13/2021", "0000-00-00", None])
            dirty_df.loc[row_idx, col] = new_value
            error_type = "Error Type 4: Invalid date format"


        elif col in ["languages"]:
            #array-like fields: introduces inconsistent data
            new_value = random.choice(["['??']", "['en', 'abc']", None])
            dirty_df.loc[row_idx, col] = new_value
            error_type = "Error Type 5: Inconsistent array-like data"

        else:
            continue

        error_log.append((row_idx, col, original_value, dirty_df.loc[row_idx, col], error_type))

        #casts integer columns to nullable Int64 to prevent float conversion
        if dirty_df[col].dtype == "int64":
            dirty_df[col] = dirty_df[col].astype("Int64")

    return dirty_df, error_log

input_file = "clean.csv"
output_file = "dirty.csv"
error_log_file = "error_log.txt"

if os.path.exists(input_file):
    try:
        df = pd.read_csv(input_file)

        num_errors = 200  #num errors introduced
        dirty_df, error_log = introduce_errors(df, num_errors)

        dirty_df.to_csv(output_file, index=False)
        print(f"Dirty data generated and saved to {output_file}.")

        with open(error_log_file, 'w') as log_file:
            for log in error_log:
                log_file.write(f"Row {log[0]}, Column '{log[1]}': Original='{log[2]}' -> New='{log[3]}', {log[4]}\n")

        print(f"\nError log saved to {error_log_file}.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
else:
    print(f"File {input_file} not found. Please ensure the file exists.")