"""
Production-ready FAQ Chatbot with Automated Learning Pipeline
Refactored for Streamlit Cloud Compatibility
------------------------------------------------------------
- Removed threading lock usage in session state.
- Simplified vector store reloading without locks.
- Updated deprecated st.experimental_rerun() to st.rerun().
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import streamlit as st
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Configuration and Constants
# ---------------------------

APP_DIR = Path.cwd()
FAQS_FILE = APP_DIR / "faqs.json"
UNRESOLVED_FILE = APP_DIR / "unresolved_queries.json"
NEGATIVE_FEEDBACK_FILE = APP_DIR / "negative_feedback.json"
NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",  # requested by spec; will attempt download (may be ignored)
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
]
SIMILARITY_THRESHOLD = 0.4
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ---------------------------
# Logging Configuration
# ---------------------------

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("faq_chatbot")

# ---------------------------
# Utility Data Classes
# ---------------------------


@dataclass
class FAQEntry:
    """Represents a single FAQ entry."""

    question: str
    answer: str


# ---------------------------
# NLTK Setup and Utilities
# ---------------------------


def ensure_nltk_resources(resources: List[str]) -> None:
    """
    Ensure required NLTK resources are downloaded.

    This function attempts to download each resource silently. If a resource
    is unavailable, it logs a warning but continues execution.
    """
    for res in resources:
        try:
            nltk.download(res, quiet=True)
            LOGGER.info("NLTK resource ensured: %s", res)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to download NLTK resource %s: %s", res, exc)


# Initialize NLTK resources at import/startup
ensure_nltk_resources(NLTK_RESOURCES)

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
    try:
        if not text:
            return ""
        text = text.lower()
        tokens = word_tokenize(text)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Tokenization failed: %s", exc)
        # Fallback: simple split
        tokens = text.split()

    try:
        pos_tags = nltk.pos_tag(tokens)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("POS tagging failed: %s", exc)
        pos_tags = [(t, "N") for t in tokens]

    processed_tokens: List[str] = []
    for token, tag in pos_tags:
        try:
            # Remove tokens that are purely punctuation
            if not any(ch.isalnum() for ch in token):
                continue
            wn_tag = _pos_tag_to_wordnet(tag)
            lemma = _LEMMATIZER.lemmatize(token, wn_tag)
            processed_tokens.append(lemma)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Lemmatization failed for token '%s': %s", token, exc)
            # Fallback: use raw token
            processed_tokens.append(token)

    return " ".join(processed_tokens)


# ---------------------------
# File Handling and Persistence
# ---------------------------


def atomic_write_json(path: Path, data: object) -> None:
    """
    Atomically write JSON to disk to reduce corruption risk.

    Args:
        path: Destination file path.
        data: JSON-serializable object.
    """
    try:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
        LOGGER.info("Wrote JSON to %s", path)
    except Exception as exc:
        LOGGER.exception("Failed to write JSON to %s: %s", path, exc)


def append_json_list(path: Path, item: object) -> None:
    """
    Append an item to a JSON list file. If file does not exist, create it.

    Args:
        path: File path.
        item: JSON-serializable item to append.
    """
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                data = []
        else:
            data = []
        data.append(item)
        atomic_write_json(path, data)
    except Exception as exc:
        LOGGER.exception("Failed to append to %s: %s", path, exc)


# ---------------------------
# Default FAQ Generation
# ---------------------------


def default_faqs() -> List[FAQEntry]:
    """
    Return a comprehensive list of default FAQs (15+ entries) for zero-config.

    These are production-ready tech/e-commerce style FAQs.
    """
    defaults = [
        FAQEntry(
            question="What is the return policy for the product?",
            answer=(
                "Our return policy allows returns within 30 days of purchase. "
                "Items must be in original condition with receipt or proof of purchase."
            ),
        ),
        FAQEntry(
            question="How do I track my order?",
            answer=(
                "You can track your order using the tracking link sent to your email "
                "after shipment. Alternatively, log in to your account and visit 'My Orders'."
            ),
        ),
        FAQEntry(
            question="Do you offer international shipping?",
            answer=(
                "Yes, we ship to many countries worldwide. Shipping fees and delivery "
                "times vary by destination."
            ),
        ),
        FAQEntry(
            question="How can I contact customer support?",
            answer=(
                "Customer support is available via email at support@example.com and "
                "via live chat on our website from 9 AM to 6 PM (local time)."
            ),
        ),
        FAQEntry(
            question="What payment methods are accepted?",
            answer=(
                "We accept major credit cards (Visa, MasterCard, American Express), "
                "PayPal, and Apple Pay."
            ),
        ),
        FAQEntry(
            question="Is there a warranty on the product?",
            answer=(
                "Yes, the product includes a one-year limited warranty covering "
                "manufacturing defects. Extended warranties are available at checkout."
            ),
        ),
        FAQEntry(
            question="How do I reset my account password?",
            answer=(
                "Click 'Forgot Password' on the sign-in page, enter your email, and "
                "follow the instructions sent to your inbox to reset your password."
            ),
        ),
        FAQEntry(
            question="Can I change or cancel my order after placing it?",
            answer=(
                "Orders can be modified or canceled within 1 hour of placement. "
                "Please contact customer support immediately for assistance."
            ),
        ),
        FAQEntry(
            question="Are there student discounts available?",
            answer=(
                "We offer a student discount program. Verify your student status via "
                "our partner portal to receive a discount code."
            ),
        ),
        FAQEntry(
            question="How do I subscribe to the newsletter?",
            answer=(
                "Subscribe by entering your email in the footer subscription box on "
                "our website. You'll receive updates, promotions, and product news."
            ),
        ),
        FAQEntry(
            question="What are the dimensions and weight of the product?",
            answer=(
                "Product dimensions and weight are listed on the product page under "
                "'Specifications'. If you need more details, contact support."
            ),
        ),
        FAQEntry(
            question="Do you provide bulk order discounts?",
            answer=(
                "Yes, bulk order discounts are available for large purchases. Contact "
                "our sales team at sales@example.com for a custom quote."
            ),
        ),
        FAQEntry(
            question="How secure is my payment information?",
            answer=(
                "We use industry-standard encryption (TLS) and do not store full card "
                "details on our servers. Payments are processed by trusted providers."
            ),
        ),
        FAQEntry(
            question="Can I change my shipping address after ordering?",
            answer=(
                "Shipping address changes are allowed within 1 hour of order placement. "
                "Contact support as soon as possible to request a change."
            ),
        ),
        FAQEntry(
            question="What is your privacy policy?",
            answer=(
                "Our privacy policy explains how we collect and use personal data. "
                "You can read it on the 'Privacy Policy' page linked in the website footer."
            ),
        ),
        FAQEntry(
            question="How long does delivery typically take?",
            answer=(
                "Delivery times vary by location and shipping method. Standard shipping "
                "usually takes 3-7 business days."
            ),
        ),
        FAQEntry(
            question="Can I return a gift item without a receipt?",
            answer=(
                "Gift returns are accepted with a gift receipt. If you do not have a "
                "receipt, contact support for assistance."
            ),
        ),
    ]
    return defaults


def ensure_faqs_file(path: Path) -> None:
    """
    Ensure faqs.json exists. If not, create it with default FAQs.

    Args:
        path: Path to faqs.json
    """
    try:
        if not path.exists():
            LOGGER.info("faqs.json not found. Creating default faqs.json.")
            faqs = default_faqs()
            data = [{"question": f.question, "answer": f.answer} for f in faqs]
            atomic_write_json(path, data)
        else:
            LOGGER.info("faqs.json found at %s", path)
    except Exception as exc:
        LOGGER.exception("Failed to ensure faqs.json: %s", exc)


def load_faqs_from_file(path: Path) -> List[FAQEntry]:
    """
    Load FAQs from faqs.json and return a list of FAQEntry.

    Args:
        path: Path to faqs.json

    Returns:
        List of FAQEntry objects.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        faqs: List[FAQEntry] = []
        for item in data:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q and a:
                faqs.append(FAQEntry(question=q, answer=a))
        LOGGER.info("Loaded %d FAQs from %s", len(faqs), path)
        return faqs
    except Exception as exc:
        LOGGER.exception("Failed to load faqs from %s: %s", path, exc)
        # Fallback to defaults to keep app running
        return default_faqs()


# ---------------------------
# Vector Store and Matching
# ---------------------------


def build_vector_store(questions: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Build TF-IDF vectorizer and matrix for a list of questions.

    Args:
        questions: List of raw question strings.

    Returns:
        A tuple of (fitted TfidfVectorizer, TF-IDF matrix).
    """
    try:
        preprocessed = [preprocess_text(q) for q in questions]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
        LOGGER.info("Vector store built with %d questions", len(questions))
        return vectorizer, tfidf_matrix
    except Exception as exc:
        LOGGER.exception("Failed to build vector store: %s", exc)
        # Return empty placeholders to avoid crashing
        empty_vectorizer = TfidfVectorizer()
        empty_matrix = np.zeros((0, 0))
        return empty_vectorizer, empty_matrix


def find_best_match(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
) -> Tuple[int, float]:
    """
    Find the best matching question index and similarity score.

    Args:
        query: Raw user query string.
        vectorizer: Fitted TfidfVectorizer.
        tfidf_matrix: TF-IDF matrix for FAQ questions.

    Returns:
        Tuple of (best_index, best_score). If no match, best_index = -1.
    """
    if not query or not query.strip():
        return -1, 0.0

    try:
        processed_query = preprocess_text(query)
    except Exception as exc:
        LOGGER.exception("Preprocessing failed for query: %s", exc)
        processed_query = query.lower()

    try:
        # If tfidf_matrix is empty, return fallback
        if tfidf_matrix.size == 0:
            return -1, 0.0
        query_vec = vectorizer.transform([processed_query])
    except Exception as exc:
        LOGGER.exception("Vectorizer transform failed: %s", exc)
        return -1, 0.0

    try:
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    except Exception as exc:
        LOGGER.exception("Cosine similarity computation failed: %s", exc)
        return -1, 0.0

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx]) if similarities.size > 0 else 0.0
    return best_idx, best_score


# ---------------------------
# Session State Management
# ---------------------------


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state keys for chat history and vector store.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Tuple[str, str]] = []

    if "faqs_mtime" not in st.session_state:
        st.session_state.faqs_mtime = 0.0

    if "faqs" not in st.session_state:
        st.session_state.faqs: List[FAQEntry] = []

    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer: Optional[TfidfVectorizer] = None

    if "tfidf_matrix" not in st.session_state:
        st.session_state.tfidf_matrix = np.zeros((0, 0))


def reload_vector_store_if_needed() -> None:
    """
    Reload faqs.json and rebuild vector store if the file has changed.

    This function is safe to call frequently; it checks file modification time
    and only rebuilds when necessary. It does not use threading locks to remain
    compatible with Streamlit cloud environments.
    """
    try:
        ensure_faqs_file(FAQS_FILE)
        try:
            current_mtime = float(FAQS_FILE.stat().st_mtime)
        except Exception as exc:
            LOGGER.exception("Failed to stat faqs.json: %s", exc)
            current_mtime = 0.0

        if current_mtime != st.session_state.faqs_mtime:
            LOGGER.info("Detected change in faqs.json; reloading vector store.")
            faqs = load_faqs_from_file(FAQS_FILE)
            st.session_state.faqs = faqs
            questions = [f.question for f in faqs]
            vectorizer, tfidf_matrix = build_vector_store(questions)
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.faqs_mtime = current_mtime
            LOGGER.info("Reload complete. FAQs loaded: %d", len(faqs))
    except Exception as exc:
        LOGGER.exception("Error while reloading vector store: %s", exc)


# ---------------------------
# Feedback and Logging
# ---------------------------


def log_unresolved_query(query: str) -> None:
    """
    Log unresolved queries (below similarity threshold) to unresolved_queries.json.

    Args:
        query: The user's raw query.
    """
    try:
        entry = {"query": query, "timestamp": datetime.utcnow().isoformat() + "Z"}
        append_json_list(UNRESOLVED_FILE, entry)
        LOGGER.info("Logged unresolved query.")
    except Exception as exc:
        LOGGER.exception("Failed to log unresolved query: %s", exc)


def log_negative_feedback(user_query: str, matched_question: str, matched_answer: str, score: float) -> None:
    """
    Log negative feedback to negative_feedback.json for administrator review.

    Args:
        user_query: The user's raw query.
        matched_question: The FAQ question that was matched.
        matched_answer: The FAQ answer that was returned.
        score: Similarity score.
    """
    try:
        entry = {
            "user_query": user_query,
            "matched_question": matched_question,
            "matched_answer": matched_answer,
            "score": score,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        append_json_list(NEGATIVE_FEEDBACK_FILE, entry)
        LOGGER.info("Logged negative feedback.")
    except Exception as exc:
        LOGGER.exception("Failed to log negative feedback: %s", exc)


# ---------------------------
# Chatbot Response Generation
# ---------------------------


def generate_bot_response(user_message: str, threshold: float = SIMILARITY_THRESHOLD) -> Tuple[str, float, Optional[int]]:
    """
    Generate a response for the user message by matching against FAQs.

    Args:
        user_message: Raw user input.
        threshold: Similarity threshold for accepting a match.

    Returns:
        Tuple of (response_text, score, matched_index). matched_index is None if no match.
    """
    if not user_message or not user_message.strip():
        return "Please enter a question so I can help.", 0.0, None

    # Ensure latest faqs and vector store
    reload_vector_store_if_needed()

    vectorizer = st.session_state.vectorizer
    tfidf_matrix = st.session_state.tfidf_matrix
    faqs = st.session_state.faqs

    if vectorizer is None or tfidf_matrix.size == 0 or not faqs:
        # Attempt to rebuild once more as self-healing
        try:
            LOGGER.warning("Vector store missing or empty; attempting rebuild.")
            faqs_local = load_faqs_from_file(FAQS_FILE)
            st.session_state.faqs = faqs_local
            questions = [f.question for f in faqs_local]
            vectorizer, tfidf_matrix = build_vector_store(questions)
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            faqs = faqs_local
        except Exception as exc:
            LOGGER.exception("Rebuild failed: %s", exc)
            return (
                "Sorry, the FAQ knowledge base is currently unavailable. Please try again later.",
                0.0,
                None,
            )

    try:
        best_idx, best_score = find_best_match(user_message, vectorizer, tfidf_matrix)
    except Exception as exc:
        LOGGER.exception("Matching failed: %s", exc)
        best_idx, best_score = -1, 0.0

    if best_idx < 0 or best_score < threshold:
        # Log unresolved query for learning pipeline
        try:
            log_unresolved_query(user_message)
        except Exception:
            LOGGER.exception("Failed to log unresolved query.")
        fallback = "I'm sorry, I don't understand. Could you rephrase your question or provide more details?"
        return fallback, best_score, None

    # Compose response with matched answer and confidence
    try:
        matched_faq = faqs[best_idx]
        confidence_pct = int(best_score * 100)
        response = (
            f"{matched_faq.answer}\n\n"
            f"_Matched FAQ:_ \"{matched_faq.question}\"  \n"
            f"_Confidence:_ {confidence_pct}%"
        )
        return response, best_score, best_idx
    except Exception as exc:
        LOGGER.exception("Failed to compose response: %s", exc)
        return "An error occurred while preparing the answer. Please try again.", best_score, None


# ---------------------------
# Streamlit UI Rendering
# ---------------------------


def render_sidebar() -> None:
    """
    Render the sidebar with metrics and session summary.
    """
    st.sidebar.header("System Status")
    try:
        reload_vector_store_if_needed()
        total_faqs = len(st.session_state.faqs)
    except Exception:
        total_faqs = 0
    st.sidebar.metric("Loaded FAQs", total_faqs)

    st.sidebar.markdown("---")
    st.sidebar.header("Session Log")
    chat_len = len(st.session_state.chat_history)
    st.sidebar.write(f"Messages this session: **{chat_len}**")
    unresolved_count = 0
    try:
        if UNRESOLVED_FILE.exists():
            with UNRESOLVED_FILE.open("r", encoding="utf-8") as fh:
                unresolved_data = json.load(fh)
            unresolved_count = len(unresolved_data)
    except Exception:
        unresolved_count = 0
    st.sidebar.write(f"Unresolved queries logged (total): **{unresolved_count}**")
    st.sidebar.markdown("---")
    st.sidebar.write("CodeAlpha AI Internship — Task 2")


def render_chat_interface() -> None:
    """
    Render the main Streamlit chat interface using st.chat_message and st.chat_input.
    """
    st.set_page_config(page_title="FAQ Chatbot (Production)", page_icon="💬", layout="centered")
    st.title("💬 FAQ Chatbot — Production")
    st.markdown(
        "Ask questions about the product and get instant answers from the embedded FAQ knowledge base. "
        "The system logs unresolved queries for continuous improvement."
    )

    render_sidebar()

    # Initialize session state and ensure faqs file exists
    initialize_session_state()
    ensure_faqs_file(FAQS_FILE)
    reload_vector_store_if_needed()

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

    # Chat input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Append and display user message
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        # Generate bot response
        response_text, score, matched_idx = generate_bot_response(user_input, threshold=SIMILARITY_THRESHOLD)

        # Append bot response to history and display
        st.session_state.chat_history.append(("assistant", response_text))
        st.chat_message("assistant").write(response_text)

        # Feedback UI: thumbs up / thumbs down
        try:
            cols = st.columns([0.1, 0.1, 1.0])
            with cols[0]:
                up_key = f"up_{len(st.session_state.chat_history)}"
                if st.button("👍", key=up_key):
                    # Simple positive feedback acknowledgement (could be extended)
                    st.success("Thanks for the feedback!")
            with cols[1]:
                down_key = f"down_{len(st.session_state.chat_history)}"
                if st.button("👎", key=down_key):
                    # Log negative feedback for admin review
                    try:
                        matched_question = st.session_state.faqs[matched_idx].question if matched_idx is not None else ""
                        matched_answer = st.session_state.faqs[matched_idx].answer if matched_idx is not None else ""
                        log_negative_feedback(user_input, matched_question, matched_answer, score)
                        st.warning("Thanks — we've recorded your feedback for review.")
                    except Exception:
                        LOGGER.exception("Failed to log negative feedback.")
                        st.error("Failed to record feedback. Please try again.")
            with cols[2]:
                st.write("")  # spacer column for layout
        except Exception:
            LOGGER.exception("Feedback UI failed to render.")

    # Provide a small control to reload faqs manually (admin convenience)
    try:
        if st.button("Reload FAQs (admin)", key="reload_faqs"):
            reload_vector_store_if_needed()
            st.rerun()
    except Exception:
        LOGGER.exception("Manual reload failed.")


def main() -> None:
    """
    Application entry point.
    """
    try:
        render_chat_interface()
    except Exception as exc:
        LOGGER.exception("Application encountered an unexpected error: %s", exc)
        st.error("The application encountered an unexpected error. Please check logs.")


if __name__ == "__main__":
    main()
