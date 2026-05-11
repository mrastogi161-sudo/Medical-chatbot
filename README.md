#  Medical Chatbot

An AI-powered medical chatbot built with LLM tool calling, chat completions, and a Gradio UI. Users can describe symptoms or ask medical questions and receive intelligent responses backed by a local SQLite medical database.

---

##  Features

- 💬 **Conversational AI** — Natural chat interface powered by an LLM
- 🔧 **Tool Calling** — LLM dynamically queries the medical database when needed
- 🗄️ **SQLite Integration** — Local medical/symptom dataset for fast lookups
- 🖥️ **Gradio UI** — Clean, browser-based chat interface
- 🔄 **Multi-turn History** — Remembers context across the conversation

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI / Gemini API |
| UI | Gradio |
| Database | SQLite |
| Language | Python |
| Data | Kaggle Medical Dataset |

---

##  Project Structure

```
Medical-chatbot/
│
├── main.py               # Main app — chat function, Gradio launch
├── tools.py              # Tool definitions and handler functions
├── database.py           # SQLite connection and query functions
├── setup_db.py           # One-time script to load CSV into SQLite
├── requirements.txt      # Dependencies
├── .env                  # API keys (never commit this!)
├── .gitignore            # Ensures .env is not pushed
└── README.md
```

---

##  Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/mrastogi161-sudo/Medical-chatbot.git
cd Medical-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Set up the database
```bash
python setup_db.py
```
This loads the medical dataset CSV into a local SQLite database.

### 5. Run the app
```bash
python main.py
```
Then open the Gradio link shown in the terminal.

---

##  How It Works

```
User types a symptom or question
        ↓
LLM decides if a database lookup is needed
        ↓
Tool call fires → SQLite queried
        ↓
Results returned to LLM
        ↓
LLM generates a natural, informative response
        ↓
Response shown in Gradio UI
```

---

##  Important

> ⚠️ This chatbot is for **informational purposes only**.  
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare provider for medical concerns.

---

##  Requirements

```
openai
gradio
python-dotenv
pandas
```

---

##  Author

**mrastogi161-sudo**  
GitHub: [@mrastogi161-sudo](https://github.com/mrastogi161-sudo)
