# AI Consciousness Research Pipeline

## Basic Idea of the model
This project model is based on research related to ai consciousness.





## Functionality of fetch_arxiv.py

### Libraries used:
arxiv to pull data from arXiv.
pandas to handle and save data in CSV.
datetime to get the current time for naming files.

### It looks for papers with any of these keywords:
"AI consciousness", "artificial intelligence consciousness", or "machine consciousness"

### arXiv search:
Max 50 results.
Sorted by most recently submitted papers (not relevance or citations)

### Extraction of useful info for each paper:

Title of the paper
Summary/abstract
When it was published and last updated
List of authors
Link to the PDF
Primary category (like cs.AI or cs.CL)
All the categories the paper belongs to


### We store everything in a list called results and then convert that list into a pandas dataframe.


### We save all data inside a data/raw folder.

If you encounter any error related to non existent directory:
the script automatically creates using os.makedirs().

### After saving, we print a success message showing how many papers were saved and the full file path




## Functionality of filter_papers.py

### Define Keywords

The script uses a predefined list of keywords to search in paper titles and summaries:

["consciousness", "sentience", "awareness", "self-awareness", "mind", "experience"]

### Load Data:

The script reads a CSV file (papers_2025-04-18.csv) located in data/raw/.

### Combine Title and Summary:

It creates a new column (text) by combining the title and summary of each paper for easier keyword searching.

### Filter Papers:

It filters the papers based on the presence of any of the defined keywords in the combined text (title + summary).

### Save Filtered Results:

The filtered papers are saved into a new CSV file (filtered_papers.csv) under the data/processed/ folder.
