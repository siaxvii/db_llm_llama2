import pandas as pd

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    columns_to_drop = ['genres', 'languages', 'networks']
    df = df.drop(columns=columns_to_drop, errors='ignore')  # 'ignore' will prevent errors if the columns are not found

    if 'weighted_rating' in df.columns:
        df = df.drop(columns=['weighted_rating'])

    df = df.head(200)

    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved to {output_file}")

input_file = "first.csv"
output_file = "clean.csv"
process_csv(input_file, output_file)