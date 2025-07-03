import io
import streamlit as st
from rag_chain_memory import main_memory as main
import logging

# Option 1: Custom HTML with specific font size
st.markdown("<h1 style='text-align: center; font-size: 30px;'>RAG based chat- based on Pre-built custom DB</h1>", 
            unsafe_allow_html=True)

# 1. Initialize memory and vectorstore only once
if "memory" not in st.session_state:
    from langchain.memory import ConversationBufferMemory
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "vs" not in st.session_state:
    from generate_db import load_vector_db
    from config import Config
    st.session_state.vs = load_vector_db(db_name=Config.DATABASE.lower())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# 2. Create sidebar for sources
with st.sidebar:
    st.header("📚 Sources")
    if st.session_state.sources:
        st.subheader("Recent Sources:")
        for i, source_group in enumerate(reversed(st.session_state.sources[-5:])):  # Show last 5 queries
            with st.expander(f"Query {len(st.session_state.sources) - i}"):
                for source in source_group:
                    st.text(f"• {source}")
    else:
        st.info("Sources will appear here after you ask questions")

# 3. Chat display logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("Ask your question here"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    try:
        # Logging setup
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger()
        logger.addHandler(handler)

        logger.info("Using cached embeddings - faster processing!")

        # 4. Pass persistent memory and vs to main
        final_answer , retrieved_docs = main(
            q=query,
            memory=st.session_state.memory,
            vs=st.session_state.vs
        )


        logging.getLogger().removeHandler(handler)
        with st.expander("Show logs"):
            st.text(log_stream.getvalue())

    except Exception as main_error:
        logging.error(f"Error in main: {str(main_error)}")
        final_answer = "Sorry, there was an error processing your request."

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    
    # Store sources for sidebar display
    current_sources = list(set([doc.metadata['source'] for doc in retrieved_docs if 'source' in doc.metadata]))
    st.session_state.sources.append(current_sources)
    
    with st.chat_message("assistant"):
        st.write(final_answer)
        
        # Optional: Still show sources inline with expander
        if current_sources:
            with st.expander("📎 View Sources"):
                for source in current_sources:
                    st.text(f"• {source}")
