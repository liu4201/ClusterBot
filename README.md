# ClusterBot
A novel RAG assistant designed to support High-Performance Computing (HPC) staff in user support roles

## Optional Arguments

### input directory
`-kb`, `--kb_dir`: Sets the path of the SupportKnowledgeBase directory. This is the path of your own SupportKnowledgeBase directory **where you put your markdown files inside**.

`-vectordb`, `--vectordb_dir`: Sets the path of the Vectordb directory. This can be a path of an empty directory.

### parameters of splitting strategies and models
`-s`, `--split_size`: Specifies the chunk size to split the document into. Default is `1500`.

`--chunk_overlap`: Sets the overlapped ratio between two chunks. Default is `0.25`.

`-m`, `--model`: Selects the LLMs to use. Default is `llama3.3:70b`. Other options are `llama-3.2-3B-Instruct`, `llama3.2:latest`, and `llama3.1:latest`.

`-e`, `--embedding`: Specifies the embedding models to use. Default is `all-mpnet-base-v2`, the other option is `all-roberta-large-v1`.

`--local`: Enables local LLMs, specifically `llama3.2-Instruct`. Default is `False`.

`--split_on_header`: Enables splitting on the header with SupportKnowledgeBase. Default is `False`.

## Installation

### Create the Conda environment:

`conda env create -f environment.yml` on Mac

`conda env create -f environment_linux.yml` on Linux



## Run ClusterBot

`conda activate robot-env`

`python server.py --local` to run ClusterBot in your machine, only for the model of Llama-3.2-3B-Instruct.

Note to run ClusterBot with other LLMs on HPC Clusters, [Purdue GenAI Studio API](https://www.rcac.purdue.edu/knowledge/genaistudio) is need. Set `AI_API_KEY=your_purdue_genai_studio_API_key`in `.env` file
