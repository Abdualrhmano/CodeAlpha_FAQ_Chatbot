
"""
Chatbot for FAQs - CodeAlpha AI Internship
------------------------------------------
Streamlit application that serves as a FAQ chatbot. The app embeds a mock
FAQ dataset, preprocesses text using NLTK, vectorizes questions with
TfidfVectorizer, and matches user queries using cosine similarity.

Features:
- In-memory FAQ dataset (no external files)
- NLTK preprocessing: lowercasing, tokenization, punctuation removal,
  lemmatization
- TF-IDF vectorization and cosine similarity matching
- Similarity threshold with polite fallback
- Streamlit chat UI using st.chat_message and st.chat_input
- Conversation history persisted in st.session_state
"""

from typing import Dict, List, Tuple

import nltk
import numpy as np
import streamlit as st
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK data is available at startup
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


def get_mock_faqs() -> Dict[str, str]:
    """
    Return an embedded dictionary of mock FAQs.

    The keys are questions and the values are answers. This dataset is
    intentionally comprehensive to allow the chatbot to work out-of-the-box.
    """
    return {
        "What is the return policy for the product?": (
            "Our return policy allows returns within 30 days of purchase. "
            "Items must be in original condition with receipt or proof of purchase."
        ),
        "How do I track my order?": (
            "You can track your order using the tracking link sent to your email "
            "after shipment. Alternatively, log in to your account and visit 'My Orders'."
        ),
        "Do you offer international shipping?": (
            "Yes, we ship to many countries worldwide. Shipping fees and delivery "
            "times vary by destination."
        ),
        "How can I contact customer support?": (
            "Customer support is available via email at support@example.com and "
            "via live chat on our website from 9 AM to 6 PM (local time)."
        ),
        "What payment methods are accepted?": (
            "We accept major credit cards (Visa, MasterCard, American Express), "
            "PayPal, and Apple Pay."
        ),
        "Is there a warranty on the product?": (
            "Yes, the product includes a one-year limited warranty covering "
            "manufacturing defects. Extended warranties are available at checkout."
        ),
        "How do I reset my account password?": (
            "Click 'Forgot Password' on the sign-in page, enter your email, and "
            "follow the instructions sent to your inbox to reset your password."
        ),
        "Can I change or cancel my order after placing it?": (
            "Orders can be modified or canceled within 1 hour of placement. "
            "Please contact customer support immediately for assistance."
        ),
        "Are there student discounts available?": (
            "We offer a student discount program. Verify your student status via "
            "our partner portal to receive a discount code."
        ),
        "How do I subscribe to the newsletter?": (
            "Subscribe by entering your email in the footer subscription box on "
            "our website. You'll receive updates, promotions, and product news."
        ),
        "What are the dimensions and weight of the product?": (
            "Product dimensions and weight are listed on the product page under "
            "'Specifications'. If you need more details, contact support."
        ),
        "Do you provide bulk order discounts?": (
            "Yes, bulk order discounts are available for large purchases. Contact "
            "our sales team at sales@example.com for a custom quote."
        ),
        "How secure is my payment information?": (
            "We use industry-standard encryption (TLS) and do not store full card "
            "details on our servers. Payments are processed by trusted providers."
        ),
        "Can I change my shipping address after ordering?": (
            "Shipping address changes are allowed within 1 hour of order placement. "
            "Contact support as soon as possible to request a change."
        ),
        "What is your privacy policy?": (
            "Our privacy policy explains how we collect and use personal data. "
            "You can read it on the 'Privacy Policy' page linked in the website footer."
        ),
    }


# Initialize lemmatizer
_LEMMATIZER = WordNetLemmatizer()


def _pos_tag_to_wordnet(tag: str) -> str:
    """
    Convert NLTK POS tag to WordNet POS tag for lemmatization.

    Args:
        tag: POS tag from nltk.pos_tag

    Returns:
        WordNet POS tag string ('n', 'v', 'a', 'r') or 'n' as default.
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text: str) -> str:
    """
    Preprocess text: lowercase, tokenize, remove punctuation, lemmatize.

    Args:
        text: Raw input string.

    Returns:
        A single string of processed tokens joined by spaces.
    """
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # POS tagging for better lemmatization
    try:
        pos_tags = nltk.pos_tag(tokens)
    except Exception:
        # Fallback: tag as nouns if POS tagging fails
        pos_tags = [(t, "N") for t in tokens]

    processed_tokens: List[str] = []
    for token, tag in pos_tags:
        # Remove punctuation and non-alphanumeric tokens
        if not any(ch.isalnum() for ch in token):
            continue
        # Lemmatize with POS
        wn_tag = _pos_tag_to_wordnet(tag)
        lemma = _LEMMATIZER.lemmatize(token, wn_tag)
        processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def build_vector_store(questions: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Build TF-IDF vectorizer and matrix for a list of questions.

    Args:
        questions: List of raw question strings.

    Returns:
        A tuple of (fitted TfidfVectorizer, TF-IDF matrix).
    """
    # Preprocess questions
    preprocessed = [preprocess_text(q) for q in questions]

    # Use TfidfVectorizer with simple token pattern (preprocessed tokens)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed)

    return vectorizer, tfidf_matrix


def find_best_match(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    questions: List[str],
) -> Tuple[int, float]:
    """
    Find the best matching question index and similarity score.

    Args:
        query: Raw user query string.
        vectorizer: Fitted TfidfVectorizer.
        tfidf_matrix: TF-IDF matrix for FAQ questions.
        questions: Original questions list (for reference).

    Returns:
        Tuple of (best_index, best_score). If no match, best_index = -1.
    """
    if not query or not query.strip():
        return -1, 0.0

    processed_query = preprocess_text(query)
    try:
        query_vec = vectorizer.transform([processed_query])
    except Exception:
        # If transform fails, return no match
        return -1, 0.0

    # Compute cosine similarity
    try:
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    except Exception:
        return -1, 0.0

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    return best_idx, best_score


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state keys for chat history and vector store.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of tuples: (role, message)

    if "faq_questions" not in st.session_state:
        faqs = get_mock_faqs()
        questions = list(faqs.keys())
        answers = list(faqs.values())
        st.session_state.faq_questions = questions
        st.session_state.faq_answers = answers

        # Build vector store and store in session state
        try:
            vectorizer, tfidf_matrix = build_vector_store(questions)
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
        except Exception as exc:
            # If building vector store fails, store empty placeholders
            st.session_state.vectorizer = None
            st.session_state.tfidf_matrix = None
            st.error("Failed to initialize FAQ vector store. See logs for details.")
            st.exception(exc)


def generate_bot_response(user_message: str, threshold: float = 0.4) -> str:
    """
    Generate a response for the user message by matching against FAQs.

    Args:
        user_message: Raw user input.
        threshold: Similarity threshold for accepting a match.

    Returns:
        Bot response string.
    """
    # Defensive checks
    if not user_message or not user_message.strip():
        return "Please enter a question so I can help."

    vectorizer = st.session_state.get("vectorizer")
    tfidf_matrix = st.session_state.get("tfidf_matrix")
    questions = st.session_state.get("faq_questions", [])
    answers = st.session_state.get("faq_answers", [])

    if vectorizer is None or tfidf_matrix is None:
        return (
            "Sorry, the FAQ knowledge base is not available at the moment. "
            "Please try again later."
        )

    try:
        best_idx, best_score = find_best_match(
            user_message, vectorizer, tfidf_matrix, questions
        )
    except Exception:
        return (
            "An error occurred while processing your question. "
            "Please try rephrasing or try again later."
        )

    # If similarity below threshold, return fallback
    if best_idx < 0 or best_score < threshold:
        return (
            "I'm sorry, I don't understand. Could you rephrase your question "
            "or provide more details?"
        )

    # Return matched answer with optional confidence note
    matched_question = questions[best_idx]
    matched_answer = answers[best_idx]
    confidence_pct = int(best_score * 100)
    response = (
        f"{matched_answer}\n\n"
        f"_Matched FAQ:_ \"{matched_question}\"  \n"
        f"_Confidence:_ {confidence_pct}%"
    )
    return response


def render_chat_interface() -> None:
    """
    Render the Streamlit chat interface using st.chat_message and st.chat_input.
    """
    st.set_page_config(
        page_title="FAQ Chatbot",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("💬 FAQ Chatbot")
    st.markdown(
        "Ask questions about the product and get instant answers from the "
        "embedded FAQ knowledge base."
    )

    # Sidebar with instructions
    with st.sidebar:
        st.header("How to use")
        st.write(
            "Type your question in the chat input below. The bot will attempt "
            "to find the best matching FAQ answer. If it cannot find a good "
            "match, it will ask you to rephrase."
        )
        st.markdown("---")
        st.write("Similarity threshold is set to **0.4** by default.")

    # Initialize session state and vector store
    initialize_session_state()

    # Display existing chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

    # Chat input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Append user message to history and display
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        # Generate bot response with robust error handling
        try:
            bot_reply = generate_bot_response(user_input, threshold=0.4)
        except Exception:
            bot_reply = (
                "An unexpected error occurred while generating a response. "
                "Please try again."
            )

        # Append bot reply to history and display
        st.session_state.chat_history.append(("assistant", bot_reply))
        st.chat_message("assistant").write(bot_reply)


def main() -> None:
    """
    Application entry point.
    """
    try:
        render_chat_interface()
    except Exception as exc:
        # Final fallback to avoid crashing the UI
        st.error("The application encountered an unexpected error.")
        st.exception(exc)


if __name__ == "__main__":
    main()
