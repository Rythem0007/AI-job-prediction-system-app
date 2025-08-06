# AI Job Prediction System

A hybrid AI-powered system that analyzes resumes against job descriptions to predict job match compatibility using both traditional machine learning and deep learning approaches.

## Features

- **Dual Model Approach**: Traditional ML (Random Forest + TF-IDF) and Deep Learning (Neural Networks with fallbacks)
- **Resume Processing**: Upload PDF and DOC/DOCX files with automatic text extraction
- **Interactive Interface**: Clean Streamlit web application with real-time analysis
- **Comprehensive Analysis**: Skill matching, confidence scoring, and detailed breakdowns
- **Visualizations**: Interactive charts and graphs showing match results

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download Project
Download all project files to a folder on your system with this structure:
```
your-project-folder/
├── .streamlit/
│   └── config.toml
├── components/
│   ├── model_comparison.py
│   └── results_display.py
├── models/
│   ├── deep_learning.py
│   └── traditional_ml.py
├── utils/
│   ├── feature_extractor.py
│   ├── pdf_processor.py
│   └── text_processor.py
├── app.py
├── requirements.txt
└── README.md
```

### Step 2: Install Required Packages
Open terminal/command prompt in your project folder and run:

```bash
pip install streamlit scikit-learn nltk pandas numpy plotly PyPDF2 python-docx torch
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
Run Python and execute:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Step 4: Run the Application
In your project folder, run:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Instructions

1. **Load Models**: Click "Load AI Models" in the sidebar (takes 1-2 minutes first time)
2. **Upload Resume**: Use the file uploader to select PDF or DOC/DOCX resume
3. **Enter Job Description**: Paste the complete job posting in the text area
4. **Analyze**: Click "Analyze Job Match" to get predictions from both models
5. **Review Results**: View detailed analysis, skill matching, and recommendations

## System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.8+
- **Internet**: Required for downloading models (first run only)

### Optional Enhancements
For advanced deep learning features, install:
```bash
pip install transformers sentence-transformers spacy
python -m spacy download en_core_web_sm
```

## Troubleshooting

### Common Issues

1. **Package Installation Errors**:
   ```bash
   pip install --upgrade pip
   pip install --user [package-name]
   ```

2. **NLTK Download Errors**:
   ```python
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context
   nltk.download('punkt')
   ```

3. **Port Already in Use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **Memory Issues**: Close other applications and restart the app

### Performance Tips
- Use PDF files for better text extraction
- Keep job descriptions under 2000 words
- Restart the app if it becomes slow

## Project Structure

- **app.py**: Main Streamlit application
- **models/**: ML model implementations
- **utils/**: Text processing and feature extraction utilities  
- **components/**: UI components for results display
- **.streamlit/**: Streamlit configuration

## Technical Details

### Traditional ML Model
- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF, skill matching, experience analysis
- **Training**: Synthetic data generation for demonstration

### Deep Learning Model  
- **Approach**: Neural networks with embeddings
- **Fallbacks**: TF-IDF when advanced libraries unavailable
- **Features**: Semantic similarity, contextual analysis

## Academic Use

This project demonstrates:
- **Machine Learning**: Classification, feature engineering, model comparison
- **Natural Language Processing**: Text preprocessing, similarity analysis
- **Web Development**: Interactive dashboard creation
- **Software Engineering**: Modular design, error handling

Perfect for B.Tech projects, demonstrating both traditional and modern AI approaches.

## Support

If you encounter issues:
1. Check Python version compatibility
2. Ensure all packages are installed
3. Verify internet connection for model downloads
4. Try running with minimal dependencies first

## License

Open source project for educational purposes.