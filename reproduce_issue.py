
import sys
import os
import shutil
import pickle
import tempfile
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Mock heavy dependencies to speed up reproduction and avoid env issues
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['PyPDF2'] = MagicMock()
sys.modules['pdfplumber'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.text_splitter'] = MagicMock()

# Now import app
import app

def reproduce():
    print("Setting up reproduction environment...")
    test_dir = tempfile.mkdtemp(prefix="repro_bug_")
    
    # Override app directories
    app.VECTOR_STORE_DIR = test_dir
    app.DOCS_DIR = test_dir
    
    doc_id = "test_doc_repro"
    text = "Some test content for embedding."
    custom_host = "https://custom-api.example.com/v1"
    
    print(f"Building vector index with custom host: {custom_host}")
    
    # Mock get_embedding_function to avoid actual API calls
    with patch('app.get_embedding_function') as mock_get_embed:
        # Mock the embedding function to return a fake embedding
        mock_embed_fn = MagicMock()
        mock_embed_fn.return_value = MagicMock() # Mock numpy array
        # Mock shape for dimension
        mock_embed_fn.return_value.shape = (1, 384) 
        mock_get_embed.return_value = mock_embed_fn
        
        # Mock faiss.IndexFlatL2
        app.faiss.IndexFlatL2.return_value = MagicMock()
        
        # Call the function under test
        app.build_vector_index(
            doc_id=doc_id, 
            text=text, 
            embedding_model_id="openai-test", 
            api_key="sk-test", 
            api_host=custom_host
        )
        
    # Check the saved pickle file
    chunks_path = os.path.join(test_dir, f"{doc_id}.pkl")
    if not os.path.exists(chunks_path):
        print("Error: Pickle file was not created.")
        return
        
    print(f"Inspecting {chunks_path}...")
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
        
    print("Pickle content keys:", data.keys())
    
    if "base_url" in data:
        print(f"Found base_url in pickle: {data['base_url']}")
        if data['base_url'] == custom_host:
            print("SUCCESS: base_url is correctly persisted.")
        else:
            print(f"FAILURE: base_url is present but incorrect. Expected {custom_host}, got {data['base_url']}")
    else:
        print("FAILURE: base_url is MISSING from pickle file.")
        print("Bug Reproduced!")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    reproduce()
