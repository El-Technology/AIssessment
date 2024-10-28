# Project Name
.

## Prerequisites
- Python 3.8 or higher
- Git

## Setup Instructions

1. **Clone the Repository**
     ```bash
   git clone https://github.com/El-Technology/AIssessment.git
   cd MATURITY-CHECK CHATBOT


2. # Environment Setup   
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment Variables
# Copy example environment file
  cp .env.example .env

# Open .env and update the required API keys

# Run the Application Start Backend Server:
  uvicorn api:app --reload

# Run the Application Start Frontend(Streamlit) Server:
  streamlit run app.py


## Database Setup
The application uses two SQLite databases:
1. Main database: Created from MaturityCheck.csv for chatbot queries
2. Conversation database: Stores chat history
These databases are automatically created when you run the application. You don't need to set them up manually.
## Viewing the Databases
To view the SQLite databases, we recommend installing DB Browser for SQLite:
VS Code extension to see the database " "SQLite Viewer""
- Windows: Download from https://sqlitebrowser.org/dl/
- Mac: brew install --cask db-browser-for-sqlite
- Linux: sudo apt-get install sqlitebrowser`