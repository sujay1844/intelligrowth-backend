pip install transformers>=4.32.0 optimum>=1.12.0
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7
pip install -q git+https://github.com/huggingface/transformers
pip install -qU langchain Faiss-gpu tiktoken sentence-transformers
pip install -qU trl Py7zr auto-gptq optimum
pip install -q rank_bm25
pip install -q PyPdf
pip install cohere openai
pip install chromadb
