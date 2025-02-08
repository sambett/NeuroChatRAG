# PD_RAG_Conversational

PD_RAG_Conversational is a Retrieval-Augmented Generation (RAG) conversational AI system designed to provide accurate and context-aware responses by leveraging retrieval-based knowledge capabilities for simple and technical purposes .
![image](https://github.com/user-attachments/assets/d78501bc-7f90-4755-80c6-9e7b59dd2796)
![image](https://github.com/user-attachments/assets/fb46949f-ad22-4625-8e88-69a6dc9c0e58)

## message for twise peeps 'mr ahmed I believe '
After installation run streamlit run streamlit_app.py , all the scripts are available from data collection script , to creating vector db , to the rag pipeline to the integration into the streamlit app .


## Features
- You can see both awnser and Retrieved context for the question !
- You can toggle between simple and technical response
- Implements RAG to enhance response accuracy
- Supports document retrieval for context-aware conversations
- Utilizes NLP techniques for better user interactions
- Scalable and modular architecture

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/RamiIbrahim2002/PD_RAG_Conversational.git
   cd PD_RAG_Conversational
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## Usage
- Start the chatbot using `main.py`.
- Provide input queries and receive responses based on the retrieved and generated knowledge.
- Customize retrieval sources and models as needed.

## Configuration
Replace .env with your api key
  
## Contributing
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit your changes
4. Push the branch and create a Pull Request

## Data
The data script + the script to scrape that data 9000 articales from PubMed is open-source :)

## License
This project is licensed under the MIT License.


