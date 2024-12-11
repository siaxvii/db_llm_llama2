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
Clean a CSV dataset with the following fields:

1. **Numerical fields** ('id', 'number_of_seasons', 'number_of_episodes', 'vote_count'): Replace invalid values like -1 or 9999 with NaN.
2. **Text fields** ('name', 'overview', 'genres', 'networks'): Fix typographical errors.
3. **Decimal fields** ('vote_average', 'popularity', 'watchlisted_rating'): Replace invalid formats (e.g., None, -5.0) with NaN.
4. **Date fields** ('first_air_date', 'last_air_date'): Fix invalid dates like '32/13/2021' or '0000-00-00', or replace with NaN.
5. **Array fields** ('languages'): Standardize to valid language codes, replacing invalid values (e.g., ['??'], ['abc']) with 'unknown'.

Return the cleaned CSV with errors corrected. 
"""

print("skibidi 4")
clean_csv(input_csv, output_csv, prompt)