import re

def parse_row(line):
    pattern = re.compile(r"Row (\d+),")
    match = pattern.match(line)
    
    if match:
        row_number = int(match.group(1))
        return row_number, line.strip()  # Return the row number and the full line
    return None

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [parse_row(line.strip()) for line in lines if parse_row(line.strip())]

def write_matched_errors(errors, output_file):
    with open(output_file, 'w') as file:
        for error in errors:
            file.write(f"{error[1]}\n")  # Write the full line of the error

known_errors = read_file("error_log.txt")
gpt_output = read_file("gpt_error_log.txt")

known_errors_dict = {row[0]: row[1] for row in known_errors}
gpt_output_dict = {row[0]: row[1] for row in gpt_output}

correctly_identified_errors = []

for row in known_errors_dict:
    if row in gpt_output_dict:
        correctly_identified_errors.append((row, gpt_output_dict[row]))

correct_count = len(correctly_identified_errors)
total_errors = len(known_errors_dict)

print(f"GPT correctly identified {correct_count} out of {total_errors} errors.")
print(f"Rows correctly identified: {[row[0] for row in correctly_identified_errors]}")

write_matched_errors(correctly_identified_errors, "correctly_identified_errors.txt")