import requests
import PyPDF2
import re
import os
import pandas as pd
import tiktoken
import time
from io import StringIO
from groq import Groq

import numpy as np


api_key='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'
client = Groq(api_key=api_key)


def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def get_pdf_files(folder_path):
    """
    Retrieve PDF files from the specified folder path with improved error handling.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
    
    Returns:
        list: List of full paths to PDF files
    """
    # Validate folder path
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    # List to store PDF file paths
    pdf_files = []
    
    # Walk through directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if file is a PDF
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    
    # Check if any PDFs were found
    if not pdf_files:
        raise ValueError(f"No PDF files found in the folder: {folder_path}")
    
    return pdf_files


def get_txt_from_pdf(pdf_files, filter_ref=False):


    data = []

    for pdf in pdf_files:
       
        try:
            with open(pdf, 'rb') as pdf_content:
          
                pdf_reader = PyPDF2.PdfReader(pdf_content)
            
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    words = page_text.split()
                    page_text_join = ' '.join(words)

                    if filter_ref:
                        page_text_join = remove_ref(page_text_join)

                    page_len = len(page_text_join)
                    div_len = page_len // 4  # Divide the page into 4 parts
                    page_parts = [page_text_join[i*div_len:(i+1)*div_len] for i in range(4)]

                    min_tokens = 40
                    for i, page_part in enumerate(page_parts):
                        if count_tokens(page_part) > min_tokens:
                            # Append the data to the list
                            data.append({
                                'file name': os.path.basename(pdf),
                                'page number': page_num + 1,
                                'page section': i+1,
                                'content': page_part,
                                'tokens': count_tokens(page_part)
                            })
        except Exception as e:
            print(f"Error processing {pdf}: {e}")

        # Create a DataFrame from the data
    df = pd.DataFrame(data)
    return df

def remove_ref(pdf_text):

    pattern = r'(REFERENCES|Acknowledgment|ACKNOWLEDGMENT)'
    match = re.search(pattern, pdf_text)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        clean_text = pdf_text[:start_index].strip()
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            '\[[\d\w]{1,3}\].+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5};','\([\d\w]{1,3}\).+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5},',
            '\([\d\w]{1,3}\).+?[\d]{3,5},','\[[\d\w]{1,3}\].+?[\d]{3,5}','[\d\w]{1,3}\).+?[\d]{3,5}\.','[\d\w]{1,3}\).+?[\d]{3,5}',
            '\([\d\w]{1,3}\).+?[\d]{3,5}','^[\w\d,\.– ;)-]+$',
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = re.sub(pattern, '', pdf_text) if len(matches) > 500 and matches.count('.') < 2 and matches.count(',') < 2 and not matches[-1].isdigit() else pdf_text

        # Split the text into lines
        lines = pdf_text.split('\n')

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = re.sub(pattern, '', lines[i]) if len(matches) > 500 and len(re.findall('\d', matches)) < 8 and len(set(matches)) > 10 and matches.count(',') < 2 and len(matches) > 20 else lines[i]

        # Join the lines back together, excluding any empty lines
        clean_text = '\n'.join([line for line in lines if line])

    return clean_text

def split_content(input_string, tokens):
    """Splits a string into chunks based on a maximum token count. """

    MAX_TOKENS = tokens
    split_strings = []
    current_string = ""
    tokens_so_far = 0

    for word in input_string.split():
        # Check if adding the next word would exceed the max token limit
        if tokens_so_far + count_tokens(word) > MAX_TOKENS:
            # If we've reached the max tokens, look for the last dot or newline in the current string
            last_dot = current_string.rfind(".")
            last_newline = current_string.rfind("\n")

            # Find the index to cut the current string
            cut_index = max(last_dot, last_newline)

            # If there's no dot or newline, we'll just cut at the max tokens
            if cut_index == -1:
                cut_index = MAX_TOKENS

            # Add the substring to the result list and reset the current string and tokens_so_far
            split_strings.append(current_string[:cut_index + 1].strip())
            current_string = current_string[cut_index + 1:].strip()
            tokens_so_far = count_tokens(current_string)

        # Add the current word to the current string and update the token count
        current_string += " " + word
        tokens_so_far += count_tokens(word)

    # Add the remaining current string to the result list
    split_strings.append(current_string.strip())

    return split_strings


def combine_section(df):
    """Merge sections, page numbers, add up content, and tokens based on the pdf name."""
    aggregated_df = df.groupby('file name').agg({
        'content': aggregate_content,
        'tokens': aggregate_tokens
    }).reset_index()

    return aggregated_df
def combine_main_SI(df):
    """Create a new column with the main part of the file name, group the DataFrame by the new column,
    and aggregate the content and tokens."""
    df['main_part'] = df['file name'].apply(extract_title)
    merged_df = df.groupby('main_part').agg({
        'content': ''.join,
        'tokens': sum
    }).reset_index()

    return merged_df.rename(columns={'main_part': 'file name'})



def aggregate_content(series):
    """Join all elements in the series with a space separator. """
    return ' '.join(series)


def aggregate_tokens(series):
    """Sum all elements in the series."""
    return series.sum()


def extract_title(file_name):
    """Extract the main part of the file name. """
    title = file_name.split('_')[0]
    return title.rstrip('.pdf')


def model_1(df):
    """Model 1 will turn text in dataframe to a summarized reaction condition table."""
    # Initialize Groq client


    response_msgs = []

    for index, row in df.iterrows():
        column1_value = row[df.columns[0]]
        column2_value = row['content']

        max_tokens = 3000
        if count_tokens(column2_value) > max_tokens:
            context_list = split_content(column2_value, max_tokens)
        else:
            context_list = [column2_value]

        answers = ''  # Collect answers from Groq
        for context in context_list:
            print("Start to analyze paper " + str(column1_value))
            user_prompt = f"""This is an experimental section on MOF synthesis from paper {column1_value}

Context:
{context}

Q: Can you summarize the following details in a table:
compound name or chemical formula (if the name is not provided), metal source, metal amount, organic linker(s),
linker amount, modulator, modulator amount or volume, solvent(s), solvent volume(s), reaction temperature,
and reaction time?

Rules:
- If any information is not provided or you are unsure, use "N/A"
- Focus on extracting experimental conditions from only the MOF synthesis
- Ignore information related to organic linker synthesis, MOF postsynthetic modification, high throughput (HT) experiment details or catalysis reactions
- If multiple conditions are provided for the same compound, use multiple rows to represent them
- If multiple units or components are provided for the same factor (e.g., g and mol for the weight, multiple linker or metals, multiple temperature and reaction time, mixed solvents, etc), include them in the same cell and separate by comma
- The table should have 11 columns, all in lowercase:
| compound name | metal source | metal amount | linker | linker amount | modulator | modulator amount or volume | solvent | solvent volume | reaction temperature | reaction time |

Respond with ONLY the table."""

        attempts = 3
        while attempts > 0:
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",  # or another available Groq model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in extracting MOF synthesis details."},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                answers_text = response.choices[0].message.content
                # Check if response is valid
                if answers_text and not answers_text.lower().startswith("i apologize"):
                    answers += '\n' + answers_text
                    break
                else:
                    raise ValueError("Invalid or apologetic response")

            except Exception as e:
                attempts -= 1
                if attempts <= 0:
                    print(f"Error: Failed to process paper {column1_value}. Skipping. (model 1)")
                    break
                print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
                time.sleep(60)

        response_msgs.append(answers)

    df = df.copy()
    df.loc[:, 'summarized'] = response_msgs
    return df

def model_2(df):
    """Model 2 identifies experiment sections and combines results"""

    response_msgs = []
    prev_paper_name = None
    total_pages = df.groupby(df.columns[0])[df.columns[1]].max()

    for _, row in df.iterrows():
        paper_name = row[df.columns[0]]
        page_number = row[df.columns[1]]

        if paper_name != prev_paper_name:
            print(f'Processing paper: {paper_name}. Total pages: {total_pages[paper_name]}')
            prev_paper_name = paper_name

        context = row['content']

        user_prompt = """I will provide a context. Determine if the section contains a comprehensive MOF synthesis with explicit reactant quantities or solvent volumes.

Examples:
1. Context: "In a 4-mL scintillation vial, the linker H2PZVDC (91.0 mg, 0.5 mmol, 1 equiv.) was dissolved in N,N-dimethylformamide (DMF) (0.6 mL) upon sonication."
   Answer: Yes

2. Context: "Synthesis and Characterization of MOFs, Abbreviations, and General Procedures."
   Answer: No

3. Context: "The design and synthesis of metal-organic frameworks (MOFs) has yielded a large number of structures"
   Answer: No

Respond with only "Yes" or "No" based on the following context:
""" + context

        attempts = 3
        while attempts > 0:
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",  # or another available Groq model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in identifying MOF synthesis sections."},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                answers = response.choices[0].message.content.strip()

                # Validate response
                if answers in ["Yes", "No"]:
                    break
                else:
                    raise ValueError("Invalid response")

            except Exception as e:
                attempts -= 1
                if attempts > 0:
                    print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 2)")
                    time.sleep(60)
                else:
                    print(f"Error: Failed to process paper {paper_name}. Skipping. (model 2)")
                    answers = "No"
                    break

        response_msgs.append(answers)

    df = df.copy()
    df.loc[:,'classification'] = response_msgs

    # Remove consecutive "No" entries
    mask_no = df["classification"].str.startswith("No")
    mask_surrounded_by_no = mask_no.shift(1, fill_value=False) & mask_no.shift(-1, fill_value=False)
    mask_to_remove = mask_no & mask_surrounded_by_no
    filtered_df = df[~mask_to_remove]

    # Combine sections and process
    combined_df = combine_main_SI(combine_section(filtered_df))
    add_table_df = model_1(combined_df)
    return add_table_df[['file name','summarized']]

   



