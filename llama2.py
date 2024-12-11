import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print("skibidi 1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
print("skibidi 2")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

print("skibidi 3")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def clean_text(text, prompt):
    print("skibidi 8")
    full_input = f"{prompt} {text}"

    print("skibidi 9")
    generated_text = text_generator(full_input, num_return_sequences=1)[0]['generated_text']
    
    print("skibidi 10")
    return generated_text.strip()

def clean_csv(input_csv, output_csv, prompt):
    print("skibidi 5")
    df = pd.read_csv(input_csv)

    print("skibidi 6")
    print(len(df.columns))
    for column in df.columns:
        print("skibidi 7")
        df[column] = df[column].apply(lambda x: clean_text(str(x), prompt))

    print("skibidi 11")
    df.to_csv(output_csv, index=False)
    print(f"Cleaned CSV saved as {output_csv}")

input_csv = "dirty.csv"
output_csv = "llama2_cleaned_data.csv" 

prompt = """
You are given a CSV dataset with errors in the following fields:

1. Numerical fields ('id', 'number_of_seasons', 'number_of_episodes', 'vote_count') may have invalid values like -1 or 9999.
2. Text fields ('name', 'overview', 'genres', 'networks') may have typos or random characters.
3. Decimal fields ('vote_average', 'popularity', 'watchlisted_rating') may have invalid formats like None or -5.0.
4. Date fields ('first_air_date', 'last_air_date') may have invalid dates like '32/13/2021' or '0000-00-00'.
5. Array-like fields ('languages') may have inconsistent values like ['??'] or ['en', 'abc'].

Please clean the text by fixing these issues, ensuring that the fields are consistent and meaningful:
- Replace invalid numbers with appropriate values or NaN.
- Correct any typographical errors in text fields.
- Ensure decimals are in valid ranges or set them to NaN.
- Replace invalid dates with valid ones or set them to NaN.
- Standardize the array-like data in 'languages' to valid language codes or set to 'unknown'.

Return the cleaned CSV dataset with the errors corrected. 
"""

print("skibidi 4")
clean_csv(input_csv, output_csv, prompt)