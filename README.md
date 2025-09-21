# TalentMatch Pro - Automated Resume Relevance Check System

## Problem Statement

Manual resume screening is a time-consuming, inconsistent, and subjective process that placement teams face when evaluating hundreds of student resumes against specific job descriptions. The traditional approach leads to:

- **Time Inefficiency**: Hours spent manually reviewing each resume
- **Inconsistent Evaluation**: Different reviewers may have varying standards
- **Human Error**: Missing qualified candidates or overlooking key skills
- **Scalability Issues**: Difficulty handling large volumes of applications
- **Lack of Actionable Feedback**: Students don't receive specific improvement suggestions

## Solution Approach

TalentMatch Pro addresses these challenges through an AI-powered, two-component system:

### Technical Architecture

**Backend (FastAPI + AI Models)**
- Advanced text parsing for PDF and DOCX files
- Semantic similarity analysis using sentence transformers
- Keyword-based skill extraction with weighted importance
- Integration with Google's Gemini AI for intelligent feedback generation
- RESTful API for scalable processing

**Frontend (Streamlit)**
- Intuitive web interface for bulk resume processing
- Interactive dashboard with visualization
- PDF report generation for stakeholders
- Real-time progress tracking

### Key Features

- **Dual Scoring Algorithm**: Combines keyword matching with semantic similarity
- **Comprehensive Skill Mapping**: Covers 25+ technical skills across programming, data science, cloud platforms, and AI/ML
- **Educational Qualification Priority**: Special weightage for B.Tech/BE requirements
- **AI-Generated Feedback**: Personalized improvement suggestions for candidates
- **Batch Processing**: Handle multiple resumes simultaneously
- **Professional Reporting**: Generate shareable PDF summaries

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- Google API Key (for AI feedback generation)

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd resume-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

4. **Run the backend API**
```bash
python app.py --api
```

The backend will be available at `http://localhost:7860`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install frontend dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure backend URL**
   - Create `.streamlit/secrets.toml`:
```toml
BACKEND_API_URL = "http://localhost:7860"
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

### Docker Deployment

**Backend:**
```bash
docker build -t resume-analyzer-backend .
docker run -p 7860:7860 -e GOOGLE_API_KEY="your-key" resume-analyzer-backend
```

**Frontend:**
```bash
cd frontend
docker build -t resume-analyzer-frontend .
docker run -p 8501:8501 resume-analyzer-frontend
```

## Usage Guide

### Single Resume Analysis

1. **Access the application** at your deployed URL or `http://localhost:8501`
2. **Navigate to "Match Resumes" tab**
3. **Upload resume files** (PDF or DOCX format)
4. **Paste job description** in the text area
5. **Click "Analyze Resumes"** to start processing
6. **View results** in the "Latest Results" tab

### Batch Processing

1. **Upload multiple resume files** using the file uploader
2. **Provide a single job description** that all resumes will be evaluated against
3. **Monitor progress** through the real-time progress bar
4. **Review comprehensive dashboard** with:
   - Summary metrics (total analyzed, average score, high-fit candidates)
   - Comparative bar chart of all candidates
   - Detailed analysis for each resume
   - Missing skills identification

### Understanding Results

**Scoring System:**
- **Score Range**: 0-100 (higher is better)
- **Verdict Categories**: 
  - High (80-100): Excellent fit
  - Medium-High (60-79): Good fit with minor gaps
  - Medium (40-59): Moderate fit, some training needed
  - Low-Medium (25-39): Significant gaps present
  - Low (0-24): Poor fit for the role

**Components:**
- **Keyword Match**: Technical skills and qualification alignment
- **Semantic Fit**: Contextual understanding of experience and role requirements
- **AI Feedback**: Specific improvement recommendations
- **Skills Gap Analysis**: Missing technical competencies

### API Usage

**Health Check:**
```bash
curl http://localhost:7860/health
```

**Analyze Resume:**
```bash
curl -X POST http://localhost:7860/analyze \
  -F "resume=@path/to/resume.pdf" \
  -F "jd_text=Your job description here"
```

## Supported Skills & Technologies

The system recognizes and evaluates:

**Programming Languages:**
- Python, SQL, VBA

**Data Science Libraries:**
- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

**Visualization & BI Tools:**
- Power BI, Tableau, Excel, Power Query, DAX

**Development Tools:**
- Jupyter, Git, Google Colab

**Cloud Platforms:**
- Azure, AWS, IaaS, PaaS

**Big Data & Analytics:**
- Spark, Hadoop

**AI/ML Specializations:**
- Machine Learning, Deep Learning, NLP, Computer Vision, Generative AI

**Education:**
- B.Tech, B.E (with high importance weighting)

## Configuration

### Skill Importance Weights

The system uses weighted scoring where different skills have varying importance:

- **Education (B.Tech/BE)**: 20% each
- **Core Programming (Python/SQL)**: 15% each
- **Visualization (Power BI/Tableau)**: 12% each
- **Data Libraries (Pandas/NumPy)**: 10%/8%
- **AI/ML Skills**: 8-10% each

### Customization

To modify skill mappings or weights:

1. **Edit `TECH_SKILLS_KEYWORDS`** dictionary in `app.py`
2. **Adjust `SKILL_IMPORTANCE_WEIGHTS`** for different prioritization
3. **Restart the backend service**

## Troubleshooting

**Common Issues:**

1. **"Could not parse resume file"**
   - Ensure file is PDF or DOCX format
   - Check file isn't corrupted or password-protected

2. **"Backend not connected"**
   - Verify backend service is running
   - Check BACKEND_API_URL configuration
   - Ensure firewall allows connections

3. **"AI feedback generation failed"**
   - Verify GOOGLE_API_KEY is set correctly
   - Check API quota and billing status
   - System falls back to structured feedback if AI unavailable

4. **Low accuracy scores**
   - Ensure resumes contain specific technical terms
   - Check job descriptions include relevant skill keywords
   - Consider adjusting skill importance weights

## Performance Specifications

- **Processing Speed**: ~2-5 seconds per resume
- **Supported Formats**: PDF, DOCX
- **Concurrent Processing**: Up to 50 resumes in batch
- **API Rate Limits**: 100 requests per hour per client
- **Memory Requirements**: ~2GB RAM for backend service

## Contributing

When contributing to this project:

1. Follow existing code structure and commenting standards
2. Test both single and batch processing workflows  
3. Update skill mappings if adding new technologies
4. Ensure Docker containers build successfully
5. Validate API endpoints return expected JSON structure

## License

This project is released under the MIT License. See LICENSE file for details.

---

**For technical support or feature requests, please open an issue in the repository.**
