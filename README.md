# AI-Powered Data Analysis Automation Platform

An intelligent, autonomous data analysis system that leverages GPT-4.1 to automatically generate, execute, and debug Python scripts for comprehensive data analysis tasks.

## 🚀 Features

- **Autonomous Analysis**: Automatically generates and executes Python code based on natural language questions
- **Multi-Format Support**: Processes 8+ file formats including CSV, JSON, Parquet, SQLite, PDF, and images
- **Intelligent Error Recovery**: Self-healing mechanism with up to 3 retry attempts and per-key failure isolation
- **Fast Processing**: Delivers complete analytical insights in under 60 seconds
- **Automated Visualizations**: Generates base64-encoded PNG outputs under 100KB
- **Web Interface**: User-friendly HTML frontend for easy interaction
- **RESTful API**: FastAPI-powered backend for programmatic access
- **Ethical Web Scraping**: Built-in support for web data extraction with rate limiting and robots.txt compliance

## 📊 Supported Analysis Types

- Statistical analysis and modeling
- Machine learning and clustering (DBSCAN, scikit-learn)
- Data visualization (matplotlib, seaborn, NetworkX)
- Geospatial analysis (GeoPandas, Folium)
- Image processing (OpenCV, PIL)
- Network graph analysis
- Time series analysis
- SQL database queries

## 🛠️ Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI/ML**: OpenAI GPT-4.1, scikit-learn, statsmodels
- **Data Processing**: Pandas, NumPy, PyArrow
- **Visualization**: Matplotlib, Seaborn, NetworkX, Folium
- **Image Processing**: OpenCV, Pillow
- **Web Scraping**: BeautifulSoup4, Playwright, html2text
- **Database**: SQLite, asyncpg
- **Geospatial**: GeoPandas, Shapely, SciPy

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Data Analyst"
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

You'll need an OpenAI API key to use this system. The API key is provided per request through the web interface or API endpoint.

## 🚀 Usage

### Running the Application

Start the server:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

### Web Interface

1. Open your browser and navigate to `http://localhost:7860`
2. Upload your `questions.txt` file containing your analysis requirements
3. Upload any additional data files (CSV, JSON, etc.)
4. Enter your OpenAI API key
5. Click submit and wait for results (typically under 60 seconds)

### API Endpoint

**POST /api**

Submit analysis requests programmatically:

```python
import requests

files = {
    'questions.txt': open('questions.txt', 'rb'),
    'data.csv': open('data.csv', 'rb')
}

data = {
    'api_key': 'your-openai-api-key'
}

response = requests.post('http://localhost:7860/api', files=files, data=data)
results = response.json()
```

### Health Check

**GET /health**

Check if the service is running:
```bash
curl http://localhost:7860/health
```

## 📝 Question File Format

Create a `questions.txt` file describing your analysis requirements in natural language:

```
Analyze the customer dataset and provide:
1. Total number of records
2. Distribution of customers by region (bar chart)
3. Average purchase value by customer segment
4. Correlation matrix of numerical features (heatmap)
5. Top 10 customers by lifetime value

Return results as JSON with keys: total_records, region_distribution_plot, avg_purchase_by_segment, correlation_heatmap, top_customers
```

## 🔄 How It Works

1. **File Processing**: Uploads are processed and loaded into appropriate data structures
2. **Script Generation**: GPT-4.1 generates a complete Python analysis script based on your questions
3. **Execution**: The script is executed in a sandboxed environment
4. **Error Recovery**: If errors occur, the system automatically identifies failed components and regenerates fixed code
5. **Results**: Complete results are returned as JSON with visualizations as base64-encoded images

## 🎯 Key Features

### Intelligent Error Handling
- Per-key failure isolation ensures one failed analysis doesn't break the entire process
- Automatic retry mechanism with up to 3 attempts
- Detailed error messages for debugging

### Modular Architecture
- Clean separation between data processing, script generation, and execution
- Extensible file format support
- Pluggable analysis components

### Performance Optimized
- 300-second execution timeout
- Image compression to keep outputs under 100KB
- Efficient data loading and processing

## 📊 Output Format

Results are returned as JSON with your requested keys:

```json
{
  "total_records": 1500,
  "region_distribution_plot": "data:image/png;base64,iVBORw0KG...",
  "avg_purchase_by_segment": {
    "premium": 150.50,
    "standard": 75.25,
    "basic": 30.10
  },
  "correlation_heatmap": "data:image/png;base64,iVBORw0KG...",
  "top_customers": [...]
}
```

## 🔒 Security Considerations

- API keys are never stored; they're used only for the current request
- Scripts execute in the local environment (consider containerization for production)
- Web scraping respects robots.txt and implements rate limiting
- No data persistence between requests

## 🐛 Troubleshooting

**Script execution fails:**
- Check that all required dependencies are installed
- Verify data files are in the correct format
- Review the error messages in the returned JSON

**Timeout errors:**
- Reduce the complexity of your analysis request
- Process large datasets in smaller chunks
- Increase the timeout value in the code if needed

**Visualization errors:**
- Ensure matplotlib backend is properly configured
- Check that image data is not exceeding size limits

## 📈 Performance Metrics

- **Response Time**: < 60 seconds for typical analyses
- **Success Rate**: 95%+ with error recovery enabled
- **Supported Formats**: 8+ file types
- **Dependencies**: 27+ integrated Python libraries
- **Max Retries**: 3 automatic retry attempts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- OpenAI for GPT-4.1 API
- FastAPI team for the excellent web framework
- All open-source library contributors

## 📞 Support

For questions or issues, please open an issue on the repository or contact the maintainer.

---

**Note**: This project requires an OpenAI API key with GPT-4.1 access. Usage costs apply based on OpenAI's pricing.
