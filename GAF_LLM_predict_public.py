# JSON Schema Builder: https://katherlab.github.io/LLMAIx/

import openai, pandas as pd, json, time, random
from typing import Optional

# Configure your API endpoint and key
openai.api_base = "API_BASE"  # Update with your server's URL
openai.api_key = "API_KEY" # Update with your API key

# Define the structured output format (JSON schema) for a strict boolean result
grammar_template = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "GAF_extract_binary",
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

response_format = {
    "type": "structural_tag",
    "structures": [{
        "name": "GAF_extract_binary",
        "schema": {
            "type": "object",
            "properties": {"Impairment": {"type": "boolean"}},
            "required": ["Impairment"],
            "additionalProperties": False
        }
    }],
    "triggers": [{"type": "always"}]
}

grammar_template_continuous = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "GAF_extract_continuous",
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

def classify(report_text: str, timeout_s=30, max_attempts=4) -> Optional[bool]:
    messages = [
      {"role":"system","content":
       "Return only a valid JSON object matching the schema. No prose, no code fences."
      },
      {"role":"user","content":
       "Decide if the person is likely to have impaired functional outcome within the next two years. "
       "Use this decision tree:"
        "|--- GAF_baseline <= 56.00"
        "|   |--- IDSC_5: Mood (sad) <= 1.50 (out of maximum 3)"
        "|   |   |--- GAF_maximum <= 76.50"
        "|   |   |   |--- class: Impaired"
        "|   |   |--- GAF_maximum >  76.50"
        "|   |   |   |--- class: Intact"
        "|   |--- IDSC_5: Mood (sad) >  1.50 (out of maximum 3)"
        "|   |   |--- FAST19: Having  good relationships with people close you less than moderate difficulty"
        "|   |   |   |--- class: Impaired"
        "|   |   |--- FAST19: Having  good relationships with people close you moderate or severe difficulty"
        "|   |   |   |--- class: Impaired"
        "|--- GAF_baseline >  56.00"
        "|   |--- GAF_maximum <= 90.50"
        "|   |   |--- FAST7: Working in the field in which you were educated no or mild difficulty"
        "|   |   |   |--- class: Intact"
        "|   |   |--- FAST7: Working in the field in which you were educated moderate or severe difficulty"
        "|   |   |   |--- class: Impaired"
        "|   |--- GAF_maximum >  90.50"
        "|   |   |--- class: Intact"
       f"{report_text}"
      }
    ]
    for attempt in range(1, max_attempts+1):
        try:
            r = openai.ChatCompletion.create(
                model="GPT-OSS-120B",
                messages=messages,
                response_format=grammar_template,
                temperature=0.0,
                request_timeout=timeout_s,
            )
            msg = r.choices[0].message
            data = msg.get("parsed") or json.loads(msg.get("content","{}"))
            return bool(data["Impairment"])
        except Exception as e:
            # Only retry on transient failures
            emsg = str(e)
            transient = any(code in emsg for code in [" 500", "InternalServerError", "timeout", "Read timed out"])
            if not transient or attempt == max_attempts:
                raise
            # jittered backoff: 1s, 2–3s, 4–6s, etc.
            sleep_s = (2 ** (attempt-1)) + random.random()
            time.sleep(sleep_s)

# Load your CSV file (assumes columns "id" and "report")
# Batch with progress persistence
df = pd.read_csv(
    "TRR_GAF_concat.csv",
    sep=";", 
    engine="python",         # more forgiving parser
    quotechar='"',           # default but make explicit
    doublequote=True,        # "" becomes a literal quote
    escapechar="\\",         # allow \" to escape quotes
    on_bad_lines="warn",
    encoding="utf-8-sig"
)
results = []
for i, row in df.iterrows():
    rid = row.get("Proband")
    text = row.get("Concatenated_Info", "")
    try:
        start = time.time()
        res = classify(text)
        dur = time.time() - start
        print(f"{i+1}/{len(df)} {rid}: OK in {dur:.2f}s → {res}")
    except Exception as e:
        print(f"{i+1}/{len(df)} {rid}: FAIL → {e}")
        res = None
    results.append(res)
    # light pacing to reduce bursts
    time.sleep(0.2)

df["Impairment"] = results
df.to_csv("FOR_DecTree_590pat_GPT-OSS-120B.csv", index=False)
