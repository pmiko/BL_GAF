# JSON Schema Builder: https://katherlab.github.io/LLMAIx/

import openai
import pandas as pd
import json
import time

# Configure your API endpoint and key
openai.api_base = "http://g19a012/v1-openai"  # Update with your server's URL
openai.api_key = "gpustack_3ae4ed9a809ac6be_c6dde0e867f4545f999d45cb0c063aff"

# Define the structured output format (JSON schema) for a strict boolean result
grammar_template = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Impairment": {
                    "type": "boolean"
                }
            },
            "required": ["Impairment"]
        }
    }
}

grammar_template_continuous = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Impairment": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["Impairment"]
        }
    }
}

# Load your CSV file (assumes columns "id" and "report")
df = pd.read_csv("BL_GAF_LLM.csv")

# Prepare a list to collect results
Impairment_results = []

# Loop over each report, call the API, and extract the boolean output
for index, row in df.iterrows():
    report_id = row["mnppsd"]
    report_text = row["Concatenated_Info"]
    grammar = grammar_template # _continuous # continuous for GAF value #.replace("condition", "Impairment") ## replace doesn't work in dictionaries. replacement strategies available, but currently unnecessary

    messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier. Your response must strictly follow the provided JSON schema "
                "and output exactly one valid JSON object with the key 'Impairment' having number between 1 and 100. " # adapt as necessary: having a boolean outcome.
                "Do not output any extra text."
            )
        },
        {
            "role": "user",
            # "content": f"Given following information, decide what level of impaired functional outcome the person described in going to express within the next two years. The impairment is defined as Global Assessment of Functioning (GAF) between 1 and 100 points. You have to decide. {report_text}"

            "content": f"Given following information, decide if the person is likely to suffer from impaired functional outcome within the next two years. The impairment is defined as Global Assessment of Functioning (GAF) equal or below 60 points. You have to decide. {report_text}"
        }
    ]
    
    # Record start time
    start_time = time.time()

    try:
        response = openai.ChatCompletion.create(
            model="llama-3.1-nemotron-70b-q4km",  # Adjust based on your server/model configuration
            messages=messages,
            response_format=grammar,
            temperature=0.0
        )

        # Compute duration
        duration = time.time() - start_time
        print(f"Report {report_id} processed in {duration:.2f} seconds.")
        
        # Get the response text and parse it as JSON
        response_text = response.choices[0].message['content']
        data = json.loads(response_text)
        Impairment = data.get("Impairment")
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error processing report {report_id} after {duration:.2f} seconds: {e}")
        Impairment = None

    Impairment_results.append(Impairment)
    
    # Sleep briefly to avoid rate limiting if necessary
    time.sleep(1)

# Add the results as a new column in the DataFrame
df["Impairment"] = Impairment_results

# Save the updated CSV to a new file
df.to_csv("test_GPT-OSS-120B.csv", index=False)
