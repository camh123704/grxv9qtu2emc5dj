# Oracle Run Instructions

## 1) Load LightRAG Server

#### A) Copy and paste the following into a terminal window

cd ~
python -m venv venv-lightrag
source venv-lightrag/bin/activate
python -m pip install -U pip
python -m pip install -U "lightrag-hku[api]"
export PATH="$HOME/.local/bin:$PATH"
which lightrag-server
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_MODEL="mistral-nemo-256k"
export LLM_MODEL="mistral-nemo-256k"
export EMBEDDING_MODEL="nomic-embed-text"
export EMBEDDING_DIM=768
export LLM_TIMEOUT=1200
export EMBEDDING_TIMEOUT=240
export MAX_ASYNC=1
export EMBEDDING_FUNC_MAX_ASYNC=2
export LLM_MODEL_KWARGS='{"options":{"num_ctx":262144,"num_predict":1200}}'
export CHUNK_SIZE=300
export CHUNK_OVERLAP_SIZE=30
lightrag-server --host 0.0.0.0 --port 9621 --working-dir ~/work/LightRAG/rag_storage

#### B) In a NEW terminal window, copy and paste the following

cd ~/work/LightRAG  
source LightRAG/bin/activate  
python -m uvicorn jhub_proxy:app --host 0.0.0.0 --port 9622

#### C) In a NEW terminal waindow, copy and paste the following

sudo apt-get install zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

#### D) Open WebUI using the following link (IMPORTANT change "cdas2" to match WIRE computer being used e.g., "lambda2")

https://icsarl.westpoint.edu/jupyter-cdas2/user/cameron.halligan/proxy/9622/webui/


## 2) Load Streamlit UI

#### A) In a NEW terminal window, copy and paste the following

cd /home/jovyan/Oracle_local  
/home/jovyan/venvs/hfmeet/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

#### B) Open Streamlit using the following link (IMPORTANT change "cdas2" to match WIRE computer being used e.g., "lambda2")

https://icsarl.westpoint.edu/jupyter-cdas2/user/cameron.halligan/proxy/8501/

## 3) Executing Pipeline

#### A) Load relevant documents in LightRAG

#### B) Upload recording through Streamlit

#### C) Click on "Run Pipeline"


## Clear LightRAG Cache (if neccessary)

TARGET="/home/jovyan/work/LightRAG/rag_storage"
rm -rf -- "$TARGET"
mkdir -p -- "$TARGET"
ls -la "$TARGET"

## Deleting LightRAG Documents