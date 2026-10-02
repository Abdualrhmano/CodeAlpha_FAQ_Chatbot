# FAQ Chatbot (Streamlit + NLP)

A lightweight FAQ chatbot built with **Streamlit** and a classic **NLP retrieval pipeline**.  
The app matches user questions against an embedded FAQ dataset using **NLTK preprocessing**, **TF-IDF vectorization**, and **cosine similarity** scoring.

## Features

- Streamlit chat interface using `st.chat_message` and `st.chat_input`
- Embedded in-memory FAQ knowledge base (no external data files required)
- NLP preprocessing pipeline:
  - lowercasing
  - tokenization
  - punctuation/non-alphanumeric filtering
  - POS-aware lemmatization with NLTK WordNet
- TF-IDF vectorization over FAQ questions
- Cosine-similarity matching between user query and FAQ vectors
- Similarity threshold fallback (`0.4` by default) when confidence is low
- Session-based chat history and vector store caching via `st.session_state`

## How It Works

1. The app loads an embedded dictionary of FAQ questions and answers.
2. FAQ questions are preprocessed and converted into TF-IDF vectors.
3. Each user message is preprocessed with the same pipeline.
4. Cosine similarity is computed between the user query vector and all FAQ vectors.
5. The highest-scoring FAQ is returned when the score is above the threshold.
6. If the score is below threshold, the app returns a polite fallback asking for rephrasing.

## Tech Stack

- Python
- Streamlit
- NLTK
- scikit-learn
- NumPy

## Requirements

Dependencies are listed in `requirements.txt`:

- `streamlit`
- `nltk`
- `scikit-learn`
- `numpy`

Python 3.9+ is recommended.

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

From the repository root:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Usage

1. Start the app.
2. Enter a question in the chat input.
3. Review the chatbot answer, matched FAQ question, and confidence percentage.
4. If the intent is unclear, rephrase your question for a better match.

## Project Structure

```text
CodeAlpha_FAQ_Chatbot/
├── app.py
├── requirements.txt
└── README.md
```

## Notes and Limitations

- The FAQ dataset is hardcoded in `app.py`; updates require code changes.
- This is retrieval-based matching, not a generative LLM chatbot.
- Matching quality depends on phrasing and FAQ coverage.
- NLTK resources are downloaded at app startup.
- Chat history is stored in Streamlit session state and is not persisted across sessions.

## Contributing

Contributions are welcome. For small improvements:

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes with clear commit messages.
4. Open a pull request describing the motivation and impact.

## License

No license file is currently present in this repository.  
If you plan to distribute or reuse this project, add an explicit `LICENSE` file.
