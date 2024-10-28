# Maturity Check Chatbot

This is a technical assessment chatbot designed to help with maturity assessments by collecting user details and answering relevant questions. It includes a **Streamlit** frontend and a **FastAPI** backend.

---


## Setup Instructions

Follow these steps to set up and run the application locally.

### 1. Clone the Repository

```bash
git clone https://github.com/El-Technology/AIssessment.git
cd MATURITY-CHECK-CHATBOT
```

### 2. Set Up the Environment
Create and Activate a Virtual Environment

```bash
python -m venv venv

```

### 3. Activate the virtual environment:

- Windows: 
```bash 
venv\Scripts\activate
```
- Unix/MacOS: 
```bash 
source venv/bin/activate
```

##### Install Dependencies: 
```bash 
pip install -r requirements.txt
```


### 4. Configure Environment Variables
Copy the example environment file to set up required API keys:

```bash 
cp .env.example .env 
```

Then, open the .env file and update the necessary API keys or other environment-specific settings.

### 5. Run the Application
Start the Backend Server
To start the FastAPI backend server, run:
```bash 
uvicorn api:app --reload
```

Start the Frontend (Streamlit) Server
In a new terminal (with the virtual environment activated), run:
```bash 
streamlit run app.py
```


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