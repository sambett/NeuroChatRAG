
![image](https://github.com/user-attachments/assets/d78501bc-7f90-4755-80c6-9e7b59dd2796)
![image](https://github.com/user-attachments/assets/fb46949f-ad22-4625-8e88-69a6dc9c0e58)

# NeuroChatRAG

## 🏆 Award-Winning Project
**Winner of the "Implémentation d'un agent conversationnel basé sur un système RAG pour un site d'information médicale sur les maladies neurologiques (SEP, Parkinson, Alzheimer, AVC) Challenge" by ARSII** at TWISE Night Challenge!

## Project Overview
This project is a fork of [PD_RAG_Conversational](https://github.com/RamiIbrahim2002/PD_RAG_Conversational), originally created as a team hackathon project. NeuroChatRAG is a Retrieval-Augmented Generation (RAG) conversational AI system designed to provide accurate and context-aware responses about neurological conditions including Multiple Sclerosis, Parkinson's, Alzheimer's, and Stroke.

## Key Features
* **Transparent Information Delivery**: View both AI answers and the retrieved medical context
* **Adjustable Complexity**: Toggle between simple explanations and technical medical responses
* **RAG Architecture**: Enhanced response accuracy through retrieval-augmented generation
* **Comprehensive Knowledge Base**: Leverages 9,000+ medical articles from PubMed
* **User-Friendly Interface**: Streamlit-based interface for easy interaction

## Technical Implementation
* **Data Collection**: Automated scraping of PubMed articles related to neurological conditions
* **Vector Database**: Efficient storage and retrieval of medical knowledge using FAISS
* **RAG Pipeline**: Sophisticated retrieval system to augment LLM responses with accurate context
* **NLP Techniques**: Domain-specific medical embeddings with PubMedBERT for accurate interpretation of medical queries
* **Modular Architecture**: Scalable design for future extensions

## Installation

### Prerequisites
Ensure you have the following installed:
* Python 3.8+
* pip
* Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/NeuroChatRAG.git
cd NeuroChatRAG
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your OpenAI API key to `.env` file:
```
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

5. Run the application:
```bash
streamlit run streamlit_app.py
```

## Pipeline Components

### Data Collection (`Pipeline/data_collection.py`)
- Fetches relevant medical articles from PubMed
- Downloads clinical guidelines from authoritative sources
- Creates organized directory structure for data storage

### Preprocessing (`Pipeline/preprocessing.py`)
- Cleans and formats medical abstracts
- Splits texts into optimized chunks for retrieval
- Implements robust error handling for processing large datasets

### RAG Pipeline (`Pipeline/rag_pipeline.py`, `Pipeline/rag_final_stable.py`)
- Integrates domain-specific medical embeddings (PubMedBERT)
- Implements retrieval mechanisms with k=5 for comprehensive context
- Supports both technical and simplified language styles
- Built with LangChain components for flexibility and extensibility

## Usage
The application provides a conversational interface where users can:
* Ask questions about neurological conditions
* View both the AI-generated answer and the source medical context
* Adjust the technical level of responses based on their background

## Data Sources
The system uses a dataset of approximately 9,000 articles from PubMed, focused on neurological conditions. The scraping scripts and data processing pipeline are included in the repository for transparency and reproducibility.

## Future Directions
* Expansion to additional neurological conditions
* Integration with medical imaging analysis
* Multi-language support for global accessibility
* Mobile application development

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
* ARSII for organizing the challenge
* TWISE Night Challenge for the platform and recognition
* Original contributors to the PD_RAG_Conversational project
* PubMed for providing access to valuable medical literature


