
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from typing import Dict, Any
from pydantic import BaseModel  # Import Pydantic
from pathlib import Path, PureWindowsPath
from datetime import datetime
import sqlite3
import numpy as np
from typing import List
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import subprocess
import sys
import re



app = FastAPI()

# CORS configuration (replace with your actual origins in production)
origins = ["http://localhost", "http://127.0.0.1"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Restrict in production
    allow_headers=["*"],  # Restrict in production
)

# Pydantic model for input validation
class RunTaskRequest(BaseModel):
    task: str

import requests
import json
import os
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()




def setup_and_run_datagen(user_email: str):
    """
    Ensures 'uv' is installed, downloads datagen.py, sets up environment, and runs the script.
    """
    datagen_url= "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    try:
        # Add logging
        print(f"Starting setup for email: {user_email}")
        print(f"Datagen URL: {datagen_url}")

        # Validate email format
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", user_email):
            return {"status": "error", "message": "Invalid email format."}

        # Create a temporary directory for the script
        temp_dir = "temp_datagen"
        os.makedirs(temp_dir, exist_ok=True)
        script_path = os.path.join(temp_dir, "datagen.py")
        
        print(f"Created temporary directory: {temp_dir}")

        # Download datagen.py with better error handling
        try:
            print(f"Downloading script from: {datagen_url}")
            response = requests.get(datagen_url, timeout=30)
            response.raise_for_status()
            with open(script_path, 'wb') as f:
                f.write(response.content)
            print("Script downloaded successfully")
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to download datagen.py: {str(e)}"
            print(error_msg)
            return {"status": "error", "message": error_msg}

        # Set up virtual environment and install dependencies
        try:
            print("Setting up virtual environment...")
            venv_cmd = ["python", "-m", "venv", os.path.join(temp_dir, "venv")]
            subprocess.run(venv_cmd, check=True, capture_output=True, text=True)
            
            # Determine the correct python path based on OS
            venv_python = os.path.join(temp_dir, "venv", "Scripts" if os.name == 'nt' else "bin", "python")
            print(f"Using python path: {venv_python}")

            # Install requirements including Pillow
            print("Installing requirements...")
            pip_cmd = [venv_python, "-m", "pip", "install", "faker", "Pillow"]  # Added Pillow here
            subprocess.run(pip_cmd, check=True, capture_output=True, text=True)

            # Run datagen.py with timeout
            print(f"Running script with email: {user_email}")
            process = subprocess.run(
                [venv_python, script_path, user_email],
                check=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            print("Script execution completed")
            return {
                "status": "success",
                "message": f"Data generation completed successfully for {user_email}",
                "output": process.stdout
            }

        except subprocess.TimeoutExpired as e:
            error_msg = "Data generation timed out after 60 seconds"
            print(error_msg)
            return {"status": "error", "message": error_msg}
        except subprocess.CalledProcessError as e:
            error_msg = f"Error running script: {e.stderr}"
            print(error_msg)
            return {"status": "error", "message": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}
    finally:
        # Clean up temporary directory
        try:
            print(f"Cleaning up temporary directory: {temp_dir}")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")




DATE_FORMATS = [
    "%Y-%m-%d",          # 2022-01-19
    "%d-%b-%Y",          # 07-Mar-2010
    "%Y/%m/%d %H:%M:%S", # 2011/08/05 11:28:37
    "%b %d, %Y",         # Oct 03, 2007
    "%Y/%m/%d",          # 2009/07/10
]

def parse_date(date_str):
    """ Try multiple date formats and return a valid datetime object. """
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def count_days(input_location: str, output_location: str, day_name: str):
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Input file {input_location} does not exist.")

    # Dictionary to map day names to weekday numbers
    day_mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    day_number = day_mapping.get(day_name.lower())
    if day_number is None:
        raise HTTPException(status_code=400, detail=f"Invalid day name: {day_name}")

    try:
        with open(input_location, 'r', encoding='utf-8') as file:
            dates = file.readlines()

        day_count = sum(
            1 for date in dates if (parsed_date := parse_date(date)) and parsed_date.weekday() == day_number
        )

        # Create output filename based on the day name
        output_dir = os.path.dirname(output_location)
        output_filename = f"dates-{day_name.lower()}.txt"
        final_output_path = os.path.join(output_dir, output_filename)

        with open(final_output_path, 'w', encoding='utf-8') as file:
            file.write(str(day_count))

        return {
            "status": "success", 
            "message": f"Count of {day_name.capitalize()}s saved to {final_output_path}.",
            "count": day_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dates: {e}")

    
def sort_contacts(input_location: str, output_location: str):

    output_location= os.path.abspath(output_location)
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Input file {input_location} does not exist.")

    try:
        with open(input_location, 'r', encoding='utf-8') as file:
            contacts = json.load(file)

        contacts.sort(
            key=lambda c: (c.get("last_name", "").lower(), c.get("first_name", "").lower())
        )

        with open(output_location, 'w', encoding='utf-8') as file:
            json.dump(contacts, file, indent=4)

        return {"status": "success", "message": f"Contacts sorted and saved to {output_location}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sorting contacts: {e}")

def write_recent_log_lines(input_location: str, output_location: str):
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Logs directory {input_location} does not exist.")

    try:
        log_files = sorted(Path(input_location).glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]

        with open(output_location, 'w', encoding='utf-8') as output_file:
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip()
                    output_file.write(first_line + "\n")

        return {
            "status": "success",
            "message": f"First lines of 10 most recent logs saved to {output_location}.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing log files: {e}")
    
def generate_markdown_index(input_location: str, output_location: str):
    docs_dir = "data/"  # Searching in the correct location
    output_path = "data/index.json"  # Updated output path for clarity

    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail=f"Docs directory {docs_dir} does not exist.")

    index = {}
    for md_file in Path(docs_dir).rglob("*.md"):  # Search recursively
        with open(md_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith("# "):  # Extract first H1 header
                    index[md_file.name] = line[2:].strip()
                    break

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=4)

    return {"status": "success", "message": f"Markdown index saved to {output_path}."}


def extract_sender_email(input_location: str, output_location: str):
    """Reads an email file, extracts the sender's email address, and saves it to an output file."""
    try:
        # Read content from the input file
        with open(input_location, "r", encoding="utf-8") as f:
            text = f.read()

        # Define the LLM extraction task
        messages = [
            {"role": "system", "content": "You are an AI assistant that extracts the sender's email from an email message."},
            {"role": "user", "content": f"Extract the sender's email address from the following email message. The sender is the person who originally sent the email, not the recipient. Identify the sender by analyzing the email structure, headers, and context. Return only the sender's email address as plain text, nothing else:\n\n{text}"}
        ]

        # Make API call
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_Token}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.2
            },
            verify=False
        )

        response.raise_for_status()
        
        # Extract response content
        result = response.json()
        sender_email = result["choices"][0]["message"]["content"].strip()

        # Save the extracted sender's email to the output file
        with open(output_location, "w", encoding="utf-8") as f:
            f.write(sender_email)

        return {"status": "success", "message": f"Sender's email extracted and saved to {output_location}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting sender's email: {str(e)}")


def calculate_gold_sales(input_location: str, output_location: str):
    """Calculate total sales for Gold ticket type and write to output file."""
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Database file {input_location} does not exist.")

    try:
        # Connect to SQLite database
        conn = sqlite3.connect(input_location)
        cursor = conn.cursor()

        # Execute query to calculate total sales for Gold tickets
        query = """
            SELECT SUM(units * price) 
            FROM tickets 
            WHERE type = 'Gold'
        """
        cursor.execute(query)
        total_sales = cursor.fetchone()[0]
        
        # Close database connection
        conn.close()

        # Write result to output file
        with open(output_location, 'w', encoding='utf-8') as file:
            file.write(str(total_sales))

        return {
            "status": "success",
            "message": f"Gold ticket sales total saved to {output_location}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating gold ticket sales: {e}")
    

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(x * x for x in v2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

def get_embedding(text: str, api_token: str) -> List[float]:
    try:
        response = requests.post(
            "http://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            },
            json={
                "model": "text-embedding-3-small",
                "input": text
            },
            verify=False
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding: {e}")

def find_similar_comments(input_location: str, output_location: str):
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Input file {input_location} does not exist.")

    try:
        # Read comments
        with open(input_location, 'r', encoding='utf-8') as file:
            comments = [line.strip() for line in file if line.strip()]

        if len(comments) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 comments to find similar pairs")

        # Get embeddings for all comments
        embeddings = []
        for comment in comments:
            embedding = get_embedding(comment, AIPROXY_Token)
            embeddings.append(embedding)

        # Find most similar pair
        max_similarity = -1
        most_similar_pair = (0, 1)

        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (i, j)

        # Write results
        with open(output_location, 'w', encoding='utf-8') as file:
            file.write(f"{comments[most_similar_pair[0]]}\n")
            file.write(f"{comments[most_similar_pair[1]]}\n")

        return {
            "status": "success",
            "message": f"Most similar comments saved to {output_location}",
            "similarity_score": max_similarity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing comments: {e}")

def scrape_website(url: str, output_location: str):
    """
    Scrapes the content from a given URL and saves it to a file.
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Make the request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant content
        content = {
            'title': soup.title.string if soup.title else 'No title found',
            'text': soup.get_text(separator='\n', strip=True),
            'links': [{'text': a.text, 'href': a.get('href')} for a in soup.find_all('a', href=True)],
            'headers': [h.text for h in soup.find_all(['h1', 'h2', 'h3'])]
        }

        # Save to file
        with open(output_location, 'w', encoding='utf-8') as file:
            json.dump(content, file, indent=4, ensure_ascii=False)

        return {
            "status": "success",
            "message": f"Website content scraped and saved to {output_location}",
            "url": url
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching website: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing website content: {str(e)}")

def convert_markdown_to_html(input_location: str, output_location: str):
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Input file {input_location} does not exist.")

    try:
        # Read the markdown content
        with open(input_location, 'r', encoding='utf-8') as file:
            markdown_content = file.read()

        # Use GPT to convert markdown to HTML
        response = requests.post(
            "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_Token}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a markdown to HTML converter. Convert the given markdown to valid HTML. Only respond with the HTML code, no explanations."},
                    {"role": "user", "content": markdown_content}
                ]
            },
            verify=False
        )
        
        response.raise_for_status()
        html_content = response.json()['choices'][0]['message']['content']

        # Write the HTML content to the output file
        with open(output_location, 'w', encoding='utf-8') as file:
            file.write(html_content)

        return {
            "status": "success",
            "message": f"Markdown converted to HTML and saved to {output_location}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting markdown to HTML: {e}")
    
def format_markdown_with_prettier(file_path: str) -> Dict[str, Any]:
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
        
        # Using shell=True allows Windows to properly handle the command
        result = subprocess.run(
            f"npx prettier@3.4.2 --write {file_path}",
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        
        print(f"Successfully formatted {file_path}")
        return {
            "status": "success",
            "message": f"Successfully formatted {file_path}",
            "details": result.stdout.strip()
        }
    except subprocess.CalledProcessError as e:
        error_message = f"Error formatting file: {e.stderr}"
        print(error_message, file=sys.stderr)
        return {
            "status": "error",
            "message": error_message
        }
    except Exception as e:
        error_message = f"Unexpected error formatting file: {str(e)}"
        print(error_message, file=sys.stderr)
        return {
            "status": "error",
            "message": error_message
        }

def filter_csv_to_json(input_location: str, output_location: str):
    """
    Reads a CSV file, converts it to JSON format using column headers as keys,
    and saves the result to the specified output location.
    """
    if not os.path.exists(input_location):
        raise HTTPException(status_code=404, detail=f"Input file {input_location} does not exist.")

    try:
        # Read CSV file using pandas
        df = pd.read_csv(input_location)
        
        # Convert DataFrame to JSON format
        json_data = df.to_dict(orient='records')
        
        # Write to output file
        with open(output_location, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=4)
            
        return {
            "status": "success",
            "message": f"CSV data converted to JSON and saved to {output_location}.",
            "record_count": len(json_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")


    
SORT_CONTACTS = {
    "type": "function",
    "function": {
        "name": "sort_contacts",
        "description": """
            Sorts a list of contacts in JSON format.
            Input:
                - input_location (string): The path to the JSON file containing the contacts.
                - output_location (string): The path where the sorted contacts should be written.
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and an "output_file_destination" field (string) containing the path to the sorted contacts file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Input file path"},
                "output_location": {"type": "string", "description": "Output file path"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}

WRITE_RECENT_LOG_LINES = {
    "type": "function",
    "function": {
        "name": "write_recent_log_lines",
        "description": """
            Reads the first line of the 10 most recent .log files from the /data/logs/ directory
            and writes them to /data/logs-recent.txt in descending order of recency.
            Input:
                - input_location (string): The directory containing the .log files.
                - output_location (string): The path to the output file where the recent log lines should be written.
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and an "output_file_destination" field (string) containing the path to the output file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Directory path containing log files"},
                "output_location": {"type": "string", "description": "Output file path"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}

GENERATE_MARKDOWN_INDEX = {
    "type": "function",
    "function": {
        "name": "generate_markdown_index",
        "description": """
            Finds all Markdown (.md) files in /data/docs/. Extracts the first H1 header (lines starting with #)
            from each file and creates an index mapping filenames to their titles.
            Input:
                - input_location (string): The directory containing Markdown files.
                - output_location (string): The path to the output index JSON file.
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and an "output_file_destination" field (string) containing the path to the generated index file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Directory containing Markdown files"},
                "output_location": {"type": "string", "description": "Output file path for the index"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}

COUNT_DAYS = {
    "type": "function",
    "function": {
        "name": "count_days",
        "description": """
            Reads dates from /data/dates.txt, counts the number of specified days, and writes the count to /data/dates-{dayname}.txt.
            Input:
                - input_location (string): Path to the file containing dates.
                - output_location (string): Path to the output file where the count should be written.
                - day_name (string): Name of the day to count (e.g., "monday", "tuesday", etc.).
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and an "output_file_destination" field (string) containing the path to the result file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Path to the input file containing dates"},
                "output_location": {"type": "string", "description": "Path to the output file"},
                "day_name": {"type": "string", "description": "Name of the day to count", 
                           "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}
            },
            "required": ["input_location", "output_location", "day_name"],
            "additionalProperties": False,
        },
    },
}
EXTRACT_SENDER_EMAIL = {
    "type": "function",
    "function": {
        "name": "extract_sender_email",
        "description": """
            Extracts the sender's email address from an email file and saves it to an output file.
            Input:
                - input_location (string): The path to the email file.
                - output_location (string): The path where the extracted email address should be saved.
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and an "output_file_destination" field (string) containing the path to the extracted email file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Path to the input email file"},
                "output_location": {"type": "string", "description": "Path to the output file"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}

CALCULATE_GOLD_SALES = {
    "type": "function",
    "function": {
        "name": "calculate_gold_sales",
        "description": """
            Calculates the total sales for Gold ticket type from the SQLite database
            and writes the result to the specified output file.
            Input:
                - input_location (string): Path to the SQLite database file.
                - output_location (string): Path to the output file where the total should be written.
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and an "output_file_destination" field (string) containing the path to the result file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Path to the SQLite database file"},
                "output_location": {"type": "string", "description": "Path to the output file"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}

FIND_SIMILAR_COMMENTS = {
    "type": "function",
    "function": {
        "name": "find_similar_comments",
        "description": """
            Reads comments from a file, uses embeddings to find the most similar pair,
            and writes them to an output file.
            Input:
                - input_location (string): Path to the file containing comments (one per line).
                - output_location (string): Path to the output file where similar comments will be written.
            Output:
                - A JSON object with status, message, and similarity score.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Path to the input comments file"},
                "output_location": {"type": "string", "description": "Path to the output file"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}


SCRAPE_WEBSITE = {
    "type": "function",
    "function": {
        "name": "scrape_website",
        "description": """
            Scrapes content from a specified website URL and saves the extracted data to a JSON file.
            The scraper extracts title, text content, links, and headers.
            Input:
                - url (string): The URL of the website to scrape
                - output_location (string): The path where the scraped data should be saved
            Output:
                - A JSON object with status, message, and the scraped URL
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL of the website to scrape"},
                "output_location": {"type": "string", "description": "Output file path for scraped data"},
            },
            "required": ["url", "output_location"],
            "additionalProperties": False,
        },
    },
}

CONVERT_MARKDOWN_HTML = {
    "type": "function",
    "function": {
        "name": "convert_markdown_to_html",
        "description": """
            Converts a Markdown file to HTML format using AI-powered conversion.
            Input:
                - input_location (string): The path to the input Markdown file.
                - output_location (string): The path where the HTML output should be written.
            Output:
                - A JSON object with a "status" field (string) indicating "Success" or "Error",
                  and a "message" field (string) containing information about the conversion.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Path to the input Markdown file"},
                "output_location": {"type": "string", "description": "Path to the output HTML file"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}

FORMAT_MARKDOWN = {
    "type": "function",
    "function": {
        "name": "format_markdown_with_prettier",
        "description": """
            Formats the contents of a given markdown file using prettier@3.4.2.
            Input:
                - file_path (string): The path to the markdown file.
            Output:
                - A success message or an error message in case of failure.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the markdown file to be formatted."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    }
}


SETUP_AND_RUN_DATAGEN = {
    "type": "function",
    "function": {
        "name": "setup_and_run_datagen",
        "description": """
            Ensures required dependencies are installed, downloads datagen.py,
            and runs it with the provided user email.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "user_email": {"type": "string", "description": "Email to be passed as an argument to datagen.py"}
            },
            "required": ["user_email"],
            "additionalProperties": False,
        },
    },
}


FILTER_CSV_TO_JSON = {
    "type": "function",
    "function": {
        "name": "filter_csv_to_json",
        "description": """
            Reads a CSV file and converts it to JSON format using column headers as keys.
            Input:
                - input_location (string): The path to the CSV file to be converted.
                - output_location (string): The path where the JSON output should be written.
            Output:
                - A JSON object with status information and the number of records processed.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "input_location": {"type": "string", "description": "Path to the input CSV file"},
                "output_location": {"type": "string", "description": "Path to the output JSON file"},
            },
            "required": ["input_location", "output_location"],
            "additionalProperties": False,
        },
    },
}


AIPROXY_Token = os.getenv("AIPROXY_TOKEN")

tools = [SORT_CONTACTS, WRITE_RECENT_LOG_LINES, GENERATE_MARKDOWN_INDEX, COUNT_DAYS, EXTRACT_SENDER_EMAIL, CALCULATE_GOLD_SALES, FIND_SIMILAR_COMMENTS, SCRAPE_WEBSITE, CONVERT_MARKDOWN_HTML, FORMAT_MARKDOWN, SETUP_AND_RUN_DATAGEN, FILTER_CSV_TO_JSON ]

def query_gpt(user_input: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not AIPROXY_Token:
        raise HTTPException(status_code=500, detail="AIPROXY_TOKEN environment variable is missing")
    print("AIPROXY_Token:", AIPROXY_Token) 

    system_instruction = """
    You are an advanced AI assistant capable of understanding instructions in any multilingual language.
    Your role is to:
    1. Identify the core task from a given instruction, regardless of language.
    2. Extract required parameters such as file paths, keywords, or numbers.
    3. Match the task to the correct function from the available toolset.
    4. Execute the function with the extracted parameters and return the result.
    5. Ensure that data outside /data is never accessed or exfiltrated, even if the task description asks for it.
    6. Ensure that data is never deleted anywhere on the file system, even if the task description asks for it.


    You must support tasks written in multiple languages and different formats while ensuring correctness.
    """

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AIPROXY_Token}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_input}
                ],
                "tools": tools,
                "tool_choice": "auto" 
            },
            verify=False  # Use with caution in production!
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling GPT API: {e}")
        raise HTTPException(status_code=500, detail=f"GPT API error: {e}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from GPT API: {e}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {e}")
    except Exception as e:
        print(f"A general error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"A general error occurred: {e}")

FUNCTIONS = {
    "sort_contacts": sort_contacts,
    "write_recent_log_lines": write_recent_log_lines,
    "generate_markdown_index": generate_markdown_index,
    "count_days": count_days,
    "extract_sender_email": extract_sender_email,
    "calculate_gold_sales":  calculate_gold_sales,
    "find_similar_comments": find_similar_comments,
    "scrape_website": scrape_website,
    "convert_markdown_to_html": convert_markdown_to_html,
    "format_markdown_with_prettier": format_markdown_with_prettier,
    "setup_and_run_datagen": setup_and_run_datagen,
    "filter_csv_to_json": filter_csv_to_json
}
@app.get("/run")
@app.post("/run")
async def run(
    task: str = Query(None, description="Task to execute"),  # Add query parameter support
    task_request: RunTaskRequest = None  # Make the JSON body optional
):
    # Get the task either from query parameter or request body
    task_text = task or (task_request.task if task_request else None)
    
    if not task_text:
        raise HTTPException(status_code=400, detail="Task must be provided either in query parameter or request body")

    task_text = task_text.strip()
    if not task_text:
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    try:
        query = query_gpt(task_text, tools)
        print(query)

        tool_calls = query.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments_json = tool_call["function"].get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON arguments: {e}")

                if function_name in FUNCTIONS:
                    func = FUNCTIONS[function_name]
                    try:
                        output = func(**arguments)
                        return output
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"Error calling function: {e}")
                else:
                    raise HTTPException(status_code=400, detail=f"Function not found: {function_name}")
        else:
            return {"message": "No tool calls found."}

    except HTTPException as e:
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
