"""app.py:
1. Streamlit frontend 
2. manages the user interface, 
3. handles PDF uploads, 
4. stores chat history, 
5. coordinates RAG retrieval and LLM calls, 
6. triggers answer verification, 
7. and renders responses to the user."""

import streamlit as st
from PIL import Image
from api_handler import (
    DEFAULT_VERIFIER_MODEL,
    format_assistant_display_markdown,
    send_query_get_response,
    verify_tutor_response,
)
from chat_gen import generate_html
from file_upload import check_and_upload_files, delete_all_knowledge_files

logo = Image.open("message.png")

# Model ids enabled for this deployment (must exist on your OpenRouter account).
DEFAULT_MODEL = "openai/gpt-5.4-mini"
MODEL_CHOICES = [
    "openai/gpt-5.4-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-opus-4.6-fast",
    "google/gemma-4-26b-a4b-it:free",
    "x-ai/grok-4.20-multi-agent",
    "x-ai/grok-4.20",
]

c1, c2 = st.columns([0.9, 3.2])

with c1:
    st.caption("")
    st.caption("")
    st.image(logo, width=120)

with c2:
    st.title("EduMentor – Semantic RAG Tutoring System")

st.markdown("## How to use:")
st.markdown(
    """
1. Enter your **OpenRouter API key**.
2. Upload one or more **PDFs** (notes, slides, textbooks, etc.).
3. Click **Upload and Attach PDFs**.
4. Ask questions about the uploaded material in the chat box.
5. EduMentor will retrieve relevant content from your PDFs and generate answers based on it.
6. If verification is enabled, each response will also be checked against:
   - Your uploaded PDFs
   - External web sources (Wikipedia & search snippets)
"""
)

api_key = st.text_input(
    label="OpenRouter API key",
    type="password",
    help="Create a key at https://openrouter.ai/keys",
)

if api_key:
    st.sidebar.header("Choose your Model:")
    model_name = st.sidebar.selectbox(
        "OpenRouter Model Id",
        options=MODEL_CHOICES,
        index=MODEL_CHOICES.index(DEFAULT_MODEL)
        if DEFAULT_MODEL in MODEL_CHOICES
        else 0,
    )
    effective_model = model_name

    st.sidebar.divider()
    enable_verify = st.sidebar.checkbox(
        "Verify answers",
        value=True,
        help="Runs two verifier calls in parallel: course PDFs vs. draft, and web evidence vs. draft.",
    )
    _verifier_options = [
        "openai/gpt-5.4-mini",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-opus-4.6-fast",
        "x-ai/grok-4.20",
    ]
    _verifier_index = (
        _verifier_options.index(DEFAULT_VERIFIER_MODEL)
        if DEFAULT_VERIFIER_MODEL in _verifier_options
        else 0
    )
    verifier_model = st.sidebar.selectbox(
        "Verifier model",
        options=_verifier_options,
        index=_verifier_index,
    )

    kb_docs = check_and_upload_files(api_key)
    st.markdown(f"**PDFs in this session:** :blue[{len(kb_docs)}] PDF(s)")
    st.divider()

    if st.sidebar.button("Delete all uploaded PDFs in this chat."):
        removed = delete_all_knowledge_files()
        if removed:
            st.sidebar.success(f"Cleared {removed} document(s).")
        else:
            st.sidebar.info("No documents in this chat.")
        st.rerun()

    if st.sidebar.button("Generate Chat History"):
        html_data = generate_html(st.session_state.messages)
        st.sidebar.download_button(
            label="Download Chat History as HTML",
            data=html_data,
            file_name="chat_history.html",
            mime="text/html",
        )

    st.subheader("Ask Away:")
    st.caption(
        "PDF text will remain in this session until you clear it or refresh the page."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                shown = format_assistant_display_markdown(
                    message["content"],
                    message.get("verification"),
                )
                st.markdown(shown, unsafe_allow_html=True)
            else:
                st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask the tutor a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="👨🏻‍🏫"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = send_query_get_response(
                    api_key,
                    st.session_state.messages,
                    kb_docs,
                    effective_model,
                )
            verification = None
            if enable_verify:
                with st.spinner("Verifying (course PDFs + web)…"):
                    verification = verify_tutor_response(
                        api_key,
                        prompt,
                        response,
                        kb_docs,
                        verifier_model=verifier_model,
                    )
            display = format_assistant_display_markdown(response, verification)
            message_placeholder.markdown(display, unsafe_allow_html=True)
            msg = {"role": "assistant", "content": response}
            if verification is not None:
                msg["verification"] = verification
            st.session_state.messages.append(msg)

else:
    st.warning(
        "Add your **OpenRouter API key** above. Get one at "
        "[openrouter.ai/keys](https://openrouter.ai/keys)."
    )
