# Exploration of Retrieval Augmented Generation (RAG)

**Track Selected:** Track A, Text RAG (Short)

**Sentence-transformer:** all-MiniLM-L6-v2

**Generator LLM:** llama-3.3-70b-instruct-awq

# Setup & Execution Guide
## 1. Install Dependencies
Ensure your virtual environment is active, then install the required packages:

pip install -r requirements.txt

## 2. Set Environment Variables
Ensure you are connected to the UTSA VPN, then set your API variables. 

### Windows (Command Prompt):

set OPENAI_API_KEY=your_key_here

set OPENAI_BASE_URL=[http://10.246.100.230/v1](http://10.246.100.230/v1)


## 3. Run the Pipeline
Execute the scripts in the following order to reproduce the results:

### Step 1: Initialize DB (Downloads & embeds 10,000 Wikipedia passages)
python data_process.py

### Step 2: Run Part 1 (10 Baseline Queries)
python run_part1.py

### Step 3: Run Part 2 (Custom Knowledge & Cross-Corpus Queries)
(Note: Ensure your 5 custom .txt files are in the /data directory first)
python run_part2.py

### Step 4: Generate Markdown Report
python generate_report.py
