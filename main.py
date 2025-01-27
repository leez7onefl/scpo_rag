import os
import base64
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import openai
import pytesseract
from pdf2image import convert_from_bytes
from github import Github
from pinecone import Pinecone

# Constants
PAGE_TITLE = "Leonard's Assistant"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Set Page Configuration
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

def load_environment_variables(env_paths):
    """Load environment variables from specified paths."""
    for folder, filename in env_paths:
        dotenv_path = find_dotenv(os.path.join(folder, filename), raise_error_if_not_found=True, usecwd=True)
        load_dotenv(dotenv_path, override=True)

def generate_openai_response(system_prompt, user_prompt, model_params, env_variables):
    """Generate a response from the OpenAI model."""
    openai.api_key = env_variables["openai_api_key"]

    try:
        response = openai.chat.completions.create(
            model=model_params['selected_model'],
            max_tokens=model_params['max_length'],
            temperature=model_params['temperature'],
            top_p=model_params['top_p'],
            frequency_penalty=model_params['frequency_penalty'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content if response.choices else "No response found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_embedding(text, model="text-embedding-3-small"):
    """Get text embedding."""
    response =  openai.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

def summarize_text(text, model="gpt-4o", max_length=512):
    """Summarize the given text."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Summarize the following text:\n\n{text}"}],
            max_tokens=max_length,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Summary error: {str(e)}"

def refine_response(text, prompt, model="gpt-4o", max_length=1024):
    """Refine the response based on the prompt."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You speak in a friendly assistant tone."},
                {"role": "user", "content": f"Refine and summarize the relevant information in order to answer this question: {prompt}\n\n{text}"}
            ],
            max_tokens=max_length,
            temperature=1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Refinement error: {str(e)}"

def extract_text_from_pdf(file_content):
    """Extract text from a PDF file."""
    images = convert_from_bytes(file_content)
    pytesseract.pytesseract.tesseract_cmd = r"./myenv/Tesseract-OCR/tesseract.exe"
    text = "".join(pytesseract.image_to_string(image) for image in images)
    return text

def chunk_text(text, max_tokens=8191):
    """Chunk text into manageable pieces."""
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0

    for word in words:
        word_length = len(word) + 1  # Account for space
        if current_length + word_length <= max_tokens:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [word], word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_and_store_documents(repo_docs, pinecone_instance):
    """Process documents and store them in Pinecone."""
    vectors = []
    progress_bar = st.progress(0)

    for doc_id, document in enumerate(repo_docs):
        extracted_text = extract_text_from_pdf(document['file_content'])
        text_chunks = chunk_text(extracted_text, max_tokens=8191)

        for i, chunk in enumerate(text_chunks):
            summary = summarize_text(chunk)
            vector = get_embedding(chunk)
            vectors.append({
                "id": f"{doc_id}_{i}",
                "values": vector,
                "metadata": {
                    "file_path": document['file_path'],
                    "summary": summary
                }
            })

        progress_bar.progress((doc_id + 1) / len(repo_docs))

    store_vectors_in_pinecone(pinecone_instance, vectors)

def retrieve_github_documents(github_token, repo_name, branch='documents', path=None):
    """Retrieve documents from a GitHub repository."""
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    contents = repo.get_contents(path or "", ref=branch)
    documents = []

    while contents:
        file_content = contents.pop(0)
        if file_content.type == 'file' and file_content.path.endswith('.pdf'):
            content_file = repo.get_contents(file_content.path, ref=branch)
            file_content_data = base64.b64decode(content_file.content)
            documents.append({"file_path": file_content.path, "file_content": file_content_data})
        elif file_content.type == 'dir':
            contents.extend(repo.get_contents(file_content.path, ref=branch))

    return documents

def initialize_pinecone(api_key, index_name):
    """Initialize a Pinecone instance."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

def is_pinecone_index_empty(index):
    """Check if a Pinecone index is empty."""
    response = index.describe_index_stats()
    return response['namespaces']['ns1']['vector_count'] == 0

def store_vectors_in_pinecone(index, vectors):
    """Store vectors in the Pinecone index."""
    index.upsert(vectors=vectors, namespace="ns1")

def query_pinecone_index(index, query_vector, top_k=3):
    """Query the Pinecone index."""
    response = index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    return response['matches']

# Load environment variables
load_environment_variables([['env', '.env']]) #load_environment_variables([['env', '.env'], ['secrets', 'keys.env']]) if local install

env_variables = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'pinecone_key': os.getenv('PINECONE_API_KEY'),
    'pinecone_index': os.getenv('PINECONE_INDEX_NAME'),
    'github_token': os.getenv('GITHUB_TOKEN'),
    'github_repo': os.getenv('GITHUB_REPO'),
    'github_branch': os.getenv('GITHUB_BRANCH', 'documents')
}

# Initialize Pinecone
pinecone_index = initialize_pinecone(env_variables['pinecone_key'], env_variables['pinecone_index'])

# Check if Pinecone index is empty and process if necessary
if is_pinecone_index_empty(pinecone_index):
    st.info("Waking up instance and loading documents...")

    with st.spinner("Retrieving and processing documents..."):
        repo_docs = retrieve_github_documents(
            env_variables['github_token'],
            env_variables['github_repo'],
            env_variables['github_branch']
        )
        st.write("Documents retrieved.")

    with st.spinner("Chunking, embedding, vectorizing, and adding metadata..."):
        process_and_store_documents(repo_docs, pinecone_index)
        st.write("Documents processed and stored.")
else:
    st.info("Documents are already processed and stored in vectorial database.")

# Clear UI before displaying chat
st.empty()

st.markdown(
    r"""
    <style>
    .stDeployButton {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.title("Leonard's AI profile 📱")
    st.image("https://media.licdn.com/dms/image/v2/D4E03AQHVJQphgpBZsA/profile-displayphoto-shrink_800_800/B4EZRQ1Vo0HsAg-/0/1736522947560?e=1743033600&v=beta&t=aOvVBtCVATAAlVg9nRWN5heYeRPXoeAMIA0w-ZdRlsY")

    with st.expander("Parameters"):
        selected_model = st.selectbox('Model', ['gpt-4o', 'o1-mini', 'gpt-3.5-turbo'], key='selected_model')
        temperature = st.slider('Creativity -/+:', min_value=0.01, max_value=1.0, value=0.8, step=0.01)
        top_p = st.slider('Words randomness -/+:', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
        freq_penalty = st.slider('Frequency Penalty -/+:', min_value=-1.99, max_value=1.99, value=0.0, step=0.01)
        max_length = st.slider('Max Length', min_value=256, max_value=8192, value=4224, step=2)

    st.button('Clear Chat History', on_click=lambda: st.session_state.update({'messages': [{"role": "assistant", "content": "Ask me anything regarding Léonard Gonzalez! 🔮"}]}))

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me anything regarding Léonard Gonzalez! 🔮"}]

# Display the chat history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message.get("content", ""))  # Use get to avoid KeyError

# Define model parameters
model_params = {
    'selected_model': selected_model,
    'temperature': temperature,
    'top_p': top_p,
    'frequency_penalty': freq_penalty,
    'max_length': max_length
}

# Chat input handling
if user_input := st.chat_input(placeholder="Enter your message"):
    # Record user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.write(f"**User:** {user_input}")

    with st.spinner("Thinking . . . "):
        # Get the embedding for the user prompt
        query_vector = get_embedding(user_input)
        
        # Query the Pinecone index
        results = query_pinecone_index(pinecone_index, query_vector)
        
        # Compile summaries from queried results
        all_summaries = "\n".join([
            f"File: {item.metadata['file_path']} - Summary: {item.metadata['summary']} - Score: {item['score']}"
            for item in results if item.metadata
        ]) if results else "No relevant information found."
        
        # Refine the response based on available summaries
        refined_response = refine_response(all_summaries, user_input)
        
        # Display the assistant's response
        st.write(f"**Assistant:** {refined_response}")

    # Append assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": refined_response})
