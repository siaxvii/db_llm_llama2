from openai import OpenAI
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

OPEN_AI_KEY = "API KEY"
client = OpenAI(api_key=OPEN_AI_KEY)

BATCH_SIZE = 200

input_csv = "dirty.csv"
output_csv = "gpt3_cleaned_data.csv"
gold_csv = "clean.csv"
error_log_file = "gpt_error_log.txt"

def get_response(input_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for cleaning CSV datasets."},
            {"role": "user", "content": input_text}
        ],
        max_tokens=1000,
        temperature=0,
    )
    response_text = response.choices[0].message.content.strip()
    return response_text

def clean_batch(batch_data, column_name, prompt_template):
    dirty_input = batch_data.to_csv(sep='\t', index=False, header=None)
    prompt = prompt_template.format(input_text=dirty_input)
    response = get_response(prompt)
    cleaned_data = [line for line in response.split("\n") if line.strip()]
    return cleaned_data

instruction_only_prompts = {
    "id": "You will be given a list of IDs. Ensure all IDs are valid integers. Replace invalid IDs (e.g., -1, non-numeric values) with NaN. Ensure each line of input corresponds to one output.\n\n{input_text}\n\nOutput only the corrected IDs as a list, one per line.",
    "name": "You will be given a list of names. Correct any typographical errors and remove extra spaces.\n\n{input_text}\n\nOutput only the corrected names as a list, one per line.",
    "overview": "You will be given a list of descriptions. Fix typographical errors and ensure proper grammar. Remove unnecessary spaces and keep the meaning intact.\n\n{input_text}\n\nOutput only the corrected descriptions as a list, one per line.",
    "number_of_seasons": "You will be given a list of numbers representing seasons. Replace any invalid or unrealistic numbers (e.g., -1, 9999) with NaN.\n\n{input_text}\n\nOutput only the corrected numbers as a list, one per line.",
    "number_of_episodes": "You will be given a list of numbers representing episodes. Replace any invalid or unrealistic numbers (e.g., -1, 9999) with NaN.\n\n{input_text}\n\nOutput only the corrected numbers as a list, one per line.",
    "first_air_date": "You will be given a list of dates. Ensure all dates are in the format YYYY-MM-DD. Replace invalid dates (e.g., 32/13/2021) with NaN.\n\n{input_text}\n\nOutput only the corrected dates as a list, one per line.",
    "last_air_date": "You will be given a list of dates. Ensure all dates are in the format YYYY-MM-DD. Replace invalid dates (e.g., 0000-00-00) with NaN.\n\n{input_text}\n\nOutput only the corrected dates as a list, one per line.",
    "vote_count": "You will be given a list of vote counts. Replace invalid values (e.g., -1, 9999, non-numeric values) with NaN.\n\n{input_text}\n\nOutput only the corrected counts as a list, one per line.",
    "vote_average": "You will be given a list of vote averages. Replace invalid values (e.g., -5.0, out-of-range values) with NaN.\n\n{input_text}\n\nOutput only the corrected averages as a list, one per line.",
    "popularity": "You are given a list of popularity scores. Replace invalid values (e.g., -1, None) with NaN. \n\n{input_text}\n\n Return one score per line.",
    "backdrop_path": "You are given a list of image paths. Ensure paths are valid and correct any typographical errors. \n\n{input_text}\n\n Return one path per line.",
    "watchlisted_rating": "You are given a list of ratings. Replace invalid values (e.g., None, -1.0) with NaN. \n\n{input_text}\n\n Ensure the ratings are within 0.0 to 10.0. Return one rating per line."
}

few_shot_prompts = {
    "id": "You will be given a list of IDs. Ensure all IDs are valid integers. Replace invalid IDs (e.g., -1, non-numeric values) with NaN. \n\n{input_text}\n\nEnsure each line of input corresponds to one output. Examples of clean data: 1, 2, 3. Don't add any extra explanations or text besides the single output, don't change anything if it is already correct. Do not include any other information in the output besides the single valid output. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "name": "You will be given a list of names. Correct any typographical errors and remove extra spaces. \n\n{input_text}\n\nExamples of clean data: 'Game of Thrones', 'Money Heist', 'Stranger Things'. Don't add any extra explanations or text besides the single output. Ensure each line corresponds to a single clean name, one per line. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "overview": "You will be given a list of descriptions. Fix typographical errors and ensure proper grammar.\n\n{input_text}\n\nExamples of clean data: 'Seven noble families fight for control of the mythical land of Westeros...', 'To carry out the biggest heist in history...'. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines. Output only the corrected descriptions as a list, one per line.",
    "number_of_seasons": "You will be given a list of numbers representing seasons. Replace any invalid or unrealistic numbers (e.g., -1, 9999) with NaN. \n\n{input_text}\n\nExamples of clean data: 8, 3, 4, Don't add any extra explanations or text besides the single output, don't change anything if it is already correct. Ensure each line correwsponds to a single valid number or NaN if invalid and don't change anything if it is already correct. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "number_of_episodes": "You will be given a list of numbers representing episodes. Replace any invalid or unrealistic numbers (e.g., -1, 9999) with NaN. \n\n{input_text}\n\nExamples of clean data: 73, 41, 34, Don't add any extra explanations or text besides the single output, don't change anything if it is already correct. Ensure each line corresponds to a single valid number or NaN if invalid. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "first_air_date": "You will be given a list of dates. Ensure all dates are in the format YYYY-MM-DD. Replace invalid dates (e.g., 32/13/2021) with NaN. If the date is missing or improperly formatted, replace it with NaN. \n\n{input_text}\n\nExamples of clean data: '2011-04-17', '2017-05-02', '2016-07-15', Ensure each line corresponds to a single valid date. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Do not omit anything",
    "last_air_date": "You will be given a list of dates. Ensure all dates are in the format YYYY-MM-DD. Replace invalid dates (e.g., 0000-00-00) with NaN. If the date is missing or improperly formatted, replace it with NaN. \n\n{input_text}\n\nExamples of clean data: '2019-05-19', '2021-12-03', '2022-07-01', Ensure each line corresponds to a single valid date. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "vote_count": "You will be given a list of vote counts. Replace invalid values (e.g., -1, non-numeric values) with NaN. \n\n{input_text}\n\nExamples of clean data: 21857, 17836, 16161, Don't add any extra explanations or text besides the single output, don't change anything if it is already correct. Ensure each line corresponds to a single valid vote count. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "vote_average": "You will be given a list of vote averages. Replace invalid values (e.g., -5.0, out-of-range values) with NaN. \n\n{input_text}\n\nExamples of clean data: 8.442, 8.257, 8.624, Don't add any extra explanations or text besides the single output, don't change anything if it is already correct. Ensure each line corresponds to a single valid average. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "popularity": "You are given a list of popularity scores. Replace invalid values (e.g., -1, None) with NaN. \n\n{input_text}\n\nExamples of clean data: 1083.917, 96.354, 185.711, Don't add any extra explanations or text besides the single output, don't change anything if it is already correct. Ensure each line corresponds to a single valid popularity score. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines.",
    "backdrop_path": "You are given a list of image paths. Ensure paths are valid and correct any typographical errors. \n\n{input_text}\n\nExamples of clean data: '/2OMB0ynKlyIenMJWI2Dy9IWT4c.jpg', '/2MaumbgBlW1NoPo3ZJO38A6v7OS.jpg', THE LINE OF OUTPUT MUST BE EQUAL TO 200. Return one path per line.",
    "watchlisted_rating": "You are given a list of ratings. Replace invalid values (e.g., None, -1.0) with NaN. Ensure the ratings are within 0.0 to 10.0. \n\n{input_text}\n\nExamples of clean data: 10.0, 7.981487690491739, 7.553375548027352, Ensure each line corresponds to a single valid rating. THE LINE OF OUTPUT MUST BE EQUAL TO 200. Ensure the number of output lines matches the number of input lines."
}

def log_changes(row_idx, column_name, original_value, cleaned_value):
    original_value_str = str(original_value).strip()
    cleaned_value_str = str(cleaned_value).strip()
    
    if original_value_str != cleaned_value_str:
        with open(error_log_file, "a") as log_file:
            log_file.write(f"Row {row_idx}, Column '{column_name}': Original='{original_value_str}' -> New='{cleaned_value_str}'\n")

def clean_dataset(df):
    for column in df.columns:
        print(f"Cleaning {column} column...")
        cleaned_column = []
        prompt_template = few_shot_prompts.get(column, "You are tasked with cleaning this column.\n\n{input_text}\n\nOutput the cleaned data, one per line.")

        for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Processing {column} batches"):
            batch_data = df[[column]].iloc[i:i + BATCH_SIZE]
            cleaned_batch = clean_batch(batch_data, column, prompt_template)

            #checks and fills missing rows
            if len(cleaned_batch) != len(batch_data):
                print(f"Warning: Mismatch in batch size for {column} at index {i}.")
                while len(cleaned_batch) < len(batch_data):
                    cleaned_batch.append("")
                cleaned_batch = cleaned_batch[:len(batch_data)]

            for idx, (original, cleaned) in enumerate(zip(batch_data[column], cleaned_batch)):
                log_changes(i + idx, column, original, cleaned)  #logs changes

            cleaned_column.extend(cleaned_batch)


        if column == 'id':
            df[column] = df[column].astype(str).str.strip()

        df[column] = cleaned_column
    return df

# Metrics computation
def compute_column_metrics(cleaned_series, gold_series, column_name):
    cleaned_series = cleaned_series.astype(str)
    gold_series = gold_series.astype(str)

    cleaned_series, gold_series = cleaned_series.align(gold_series, join='inner')

    accuracy = accuracy_score(gold_series, cleaned_series)
    precision = precision_score(gold_series, cleaned_series, average='micro', zero_division=1)
    recall = recall_score(gold_series, cleaned_series, average='micro', zero_division=1)
    f1 = f1_score(gold_series, cleaned_series, average='micro', zero_division=1)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

def save_metrics_to_csv(metrics, filename):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


def generate_metrics_graphs(metrics, filename):
    metrics_df = pd.DataFrame(metrics)
    
    metrics_df.set_index('Column', inplace=True)
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Metrics Comparison')
    plt.ylabel('Scores')
    plt.xlabel('Columns')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.show()
    print(f"Metrics graph saved to {filename}")

if __name__ == "__main__":
    df = pd.read_csv(input_csv)
    gold_df = pd.read_csv(gold_csv)

    cleaned_df = clean_dataset(df)

    cleaned_df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

    metrics_list = []
    print("Column-wise Metrics:")
    for column in df.columns:
        metrics = compute_column_metrics(cleaned_df[column], gold_df[column], column)
        print(f"Metrics for {column}:")
        print(metrics)
        
        metrics["Column"] = column
        metrics_list.append(metrics)

    save_metrics_to_csv(metrics_list, "metrics.csv")
    generate_metrics_graphs(metrics_list, "metrics_graph.png")