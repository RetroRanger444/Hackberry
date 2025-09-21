# ==============================================================================
# Enhanced Resume Analyzer - FastAPI Implementation for Production
# ==============================================================================
import os
import re
import json
import fitz  # PyMuPDF
from docx import Document
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import Counter
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
from typing import List, Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. ENHANCED MODEL CONFIGURATION ---
print("Loading SentenceTransformer model...")
try:
    # Using a more specialized model for better semantic understanding
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load model: {e}")
    model = None

# Configure Gemini API with environment variable support
gemini_api_configured = False
try:
    # Try multiple environment variable names for flexibility
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Gemini API configured successfully.")
        gemini_api_configured = True
    else:
        print("Warning: Gemini API key not found. Set GOOGLE_API_KEY environment variable.")
except Exception as e:
    print(f"Gemini API configuration failed: {e}")

# --- 2. COMPREHENSIVE SKILLS AND ROLE MAPPING ---
TECH_SKILLS_KEYWORDS = {
    # Core Programming Languages
    'python': ['python', 'py', 'python3', 'pandas', 'numpy', 'matplotlib'],
    'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle', 'tsql', 'plsql'],
    'r': ['r programming', 'r language', 'ggplot', 'dplyr'],
    'java': ['java', 'spring', 'hibernate'],
    'javascript': ['javascript', 'js', 'node.js', 'react', 'angular'],
    
    # Data Science & Analytics
    'data analysis': ['data analysis', 'data analyst', 'statistical analysis'],
    'machine learning': ['machine learning', 'ml', 'scikit-learn', 'algorithms'],
    'deep learning': ['deep learning', 'neural networks', 'tensorflow', 'pytorch', 'keras'],
    'statistics': ['statistics', 'statistical', 'hypothesis testing', 'regression'],
    
    # Visualization & BI Tools
    'power bi': ['power bi', 'powerbi', 'power_bi', 'microsoft power bi'],
    'tableau': ['tableau', 'tableau desktop', 'tableau public'],
    'excel': ['excel', 'microsoft excel', 'ms excel', 'pivot tables'],
    'qlik': ['qlik', 'qlikview', 'qliksense'],
    
    # Cloud & Big Data
    'azure': ['azure', 'microsoft azure', 'azure data factory'],
    'aws': ['aws', 'amazon web services', 'ec2', 's3'],
    'spark': ['spark', 'apache spark', 'pyspark'],
    'hadoop': ['hadoop', 'hdfs', 'hive'],
    
    # Education Qualifications
    'btech': ['b.tech', 'btech', 'bachelor of technology'],
    'be': ['b.e', 'be', 'bachelor of engineering'],
    'mtech': ['m.tech', 'mtech', 'master of technology'],
    'mca': ['mca', 'master of computer applications'],
    'bca': ['bca', 'bachelor of computer applications'],
    
    # Non-technical skills (for mismatch detection)
    'cooking': ['cooking', 'culinary', 'chef', 'kitchen', 'food preparation'],
    'hospitality': ['hospitality', 'hotel', 'restaurant', 'customer service'],
    'sales': ['sales', 'marketing', 'crm', 'lead generation'],
    'accounting': ['accounting', 'bookkeeping', 'financial', 'audit']
}

# Role-specific skill requirements
ROLE_SKILL_MAPPING = {
    'data science': ['python', 'sql', 'machine learning', 'statistics', 'data analysis'],
    'data analyst': ['sql', 'excel', 'power bi', 'tableau', 'data analysis'],
    'business analyst': ['sql', 'excel', 'power bi', 'data analysis'],
    'ml engineer': ['python', 'machine learning', 'deep learning', 'statistics'],
    'software developer': ['python', 'java', 'javascript', 'sql'],
    'chef': ['cooking', 'culinary', 'hospitality'],
    'sales': ['sales', 'crm', 'marketing'],
}

# --- 3. ENHANCED TEXT PROCESSING ---
def normalize_text(text):
    """Advanced text normalization for better analysis."""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Fix common formatting issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)
    
    # Normalize technical terms
    replacements = {
        r'\bB\.E\b': 'BE',
        r'\bB\.Tech\b': 'BTech',
        r'\bM\.Tech\b': 'MTech',
        r'\bPower\s*BI\b': 'Power BI',
        r'\bMachine\s*Learning\b': 'Machine Learning',
        r'\bDeep\s*Learning\b': 'Deep Learning',
        r'\bData\s*Science\b': 'Data Science',
        r'\bData\s*Analyst\b': 'Data Analyst',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()

def parse_file(file_path):
    """Enhanced file parsing with better error handling."""
    try:
        _, extension = os.path.splitext(file_path)
        text = ""
        
        if extension.lower() == '.pdf':
            with fitz.open(file_path) as doc:
                for page in doc:
                    try:
                        page_text = page.get_text()
                        text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error reading PDF page: {e}")
        elif extension.lower() == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return None
            
        text = normalize_text(text)
        return text if text.strip() else None
        
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        return None

# --- 4. ENHANCED JD ANALYSIS WITH LLM ---
async def analyze_jd_with_llm(jd_text):
    """Use LLM to extract comprehensive job information."""
    if not gemini_api_configured:
        return analyze_jd_fallback(jd_text)
    
    try:
        prompt = f"""
        Analyze this job description and extract the following information in JSON format:
        
        Job Description:
        {jd_text[:2000]}  # Limit to avoid token limits
        
        Extract:
        1. job_title - The exact job title
        2. must_have_skills - List of required technical skills
        3. nice_to_have_skills - List of preferred skills
        4. qualifications - Educational requirements
        5. experience_level - Required experience (fresher/junior/mid/senior)
        6. role_category - Type of role (data science, software development, etc.)
        7. location - Job location if mentioned
        
        Return valid JSON only.
        """
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        
        try:
            # Clean the response to extract JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            jd_info = json.loads(response_text)
            
            # Normalize the extracted skills to match our keywords
            normalized_skills = []
            for skill in jd_info.get('must_have_skills', []):
                normalized_skill = normalize_skill_name(skill.lower())
                if normalized_skill:
                    normalized_skills.append(normalized_skill)
            
            jd_info['must_have_skills'] = normalized_skills
            return jd_info
            
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON, using fallback")
            return analyze_jd_fallback(jd_text)
            
    except Exception as e:
        logger.error(f"LLM JD analysis failed: {e}")
        return analyze_jd_fallback(jd_text)

def normalize_skill_name(skill_text):
    """Normalize extracted skill to our standard skill names."""
    skill_lower = skill_text.lower()
    
    for standard_skill, variations in TECH_SKILLS_KEYWORDS.items():
        for variation in variations:
            if variation.lower() in skill_lower or skill_lower in variation.lower():
                return standard_skill
    
    # Direct mapping for common variations
    direct_mapping = {
        'python programming': 'python',
        'sql database': 'sql',
        'data visualization': 'power bi',
        'microsoft excel': 'excel',
        'statistical analysis': 'statistics',
        'artificial intelligence': 'machine learning',
    }
    
    return direct_mapping.get(skill_lower, skill_text if len(skill_text) < 50 else None)

def analyze_jd_fallback(jd_text):
    """Fallback JD analysis when LLM is not available."""
    info = {
        'job_title': 'Unknown Position',
        'must_have_skills': [],
        'nice_to_have_skills': [],
        'qualifications': [],
        'experience_level': 'Not specified',
        'role_category': 'general',
        'location': 'Not specified'
    }
    
    if not jd_text:
        return info
    
    jd_lower = jd_text.lower()
    
    # Extract job title with better patterns
    title_patterns = [
        r'(?i)job title:\s*(.*?)(?:\n|$)',
        r'(?i)position:\s*(.*?)(?:\n|$)',
        r'(?i)role:\s*(.*?)(?:\n|$)',
        r'(?i)(data\s+scientist?|data\s+analyst|business\s+analyst|ml\s+engineer|software\s+developer|chef|sales\s+manager)',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, jd_text, re.MULTILINE)
        if match:
            title = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
            if title and len(title) < 100:
                info['job_title'] = title
                break
    
    # Determine role category
    if any(term in jd_lower for term in ['data scien', 'machine learn', 'ml engineer']):
        info['role_category'] = 'data science'
        info['must_have_skills'] = ['python', 'sql', 'machine learning', 'statistics']
    elif any(term in jd_lower for term in ['data analyst', 'business analyst']):
        info['role_category'] = 'data analyst'
        info['must_have_skills'] = ['sql', 'excel', 'power bi', 'data analysis']
    elif any(term in jd_lower for term in ['chef', 'cook', 'culinary', 'kitchen']):
        info['role_category'] = 'culinary'
        info['must_have_skills'] = ['cooking', 'culinary', 'hospitality']
    elif any(term in jd_lower for term in ['sales', 'marketing', 'business development']):
        info['role_category'] = 'sales'
        info['must_have_skills'] = ['sales', 'crm', 'marketing']
    
    # Skill extraction
    found_skills = set()
    for skill_category, variations in TECH_SKILLS_KEYWORDS.items():
        for variation in variations:
            if re.search(r'\b' + re.escape(variation) + r'\b', jd_lower):
                found_skills.add(skill_category)
                break
    
    if found_skills:
        info['must_have_skills'] = list(found_skills)
    
    return info

# --- 5. ENHANCED SKILL EXTRACTION ---
def extract_resume_skills_comprehensive(resume_text):
    """Comprehensive skill extraction with context awareness."""
    if not resume_text:
        return set(), ""
    
    resume_lower = resume_text.lower()
    found_skills = set()
    resume_category = "general"
    
    # Determine resume category first
    if any(term in resume_lower for term in ['data scien', 'machine learn', 'python', 'sql', 'analytics']):
        resume_category = "technical"
    elif any(term in resume_lower for term in ['chef', 'cook', 'culinary', 'kitchen', 'restaurant']):
        resume_category = "culinary"
    elif any(term in resume_lower for term in ['sales', 'marketing', 'crm', 'business development']):
        resume_category = "sales"
    
    # Extract skills based on keywords
    for skill_category, variations in TECH_SKILLS_KEYWORDS.items():
        for variation in variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, resume_lower):
                found_skills.add(skill_category)
                break
    
    # Section-based extraction
    sections = {
        'skills': r'(?i)skills?\s*:?\s*(.*?)(?:\n\n|\n[A-Z]|experience|education|projects)',
        'projects': r'(?i)projects?\s*:?\s*(.*?)(?:\n\n|\n[A-Z]|experience|education)',
        'experience': r'(?i)experience\s*:?\s*(.*?)(?:\n\n|\n[A-Z]|education|projects)',
    }
    
    for section_name, pattern in sections.items():
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            section_text = match.group(1).lower()
            for skill_category, variations in TECH_SKILLS_KEYWORDS.items():
                if skill_category not in found_skills:
                    for variation in variations:
                        if variation in section_text:
                            found_skills.add(skill_category)
                            break
    
    return found_skills, resume_category

# --- 6. ENHANCED SEMANTIC MATCHING WITH LLM ---
async def calculate_semantic_relevance_with_llm(resume_text, jd_text, jd_info):
    """Use LLM for semantic relevance scoring."""
    if not gemini_api_configured:
        return calculate_semantic_fallback(resume_text, jd_text)
    
    try:
        prompt = f"""
        You are an expert resume evaluator. Analyze how well this resume matches the job requirements.
        
        Job Title: {jd_info.get('job_title', 'Unknown')}
        Job Category: {jd_info.get('role_category', 'general')}
        Required Skills: {', '.join(jd_info.get('must_have_skills', []))}
        
        Resume Content (first 1000 chars):
        {resume_text[:1000]}
        
        Evaluate:
        1. How well does the candidate's experience match the role?
        2. Are there any major red flags (e.g., chef applying for data science)?
        3. Does the context and language suggest domain expertise?
        
        Respond with a JSON object containing:
        - "relevance_score": float between 0.0 and 1.0
        - "domain_match": boolean (true if domains align, false for major mismatches)
        - "explanation": brief explanation of the score
        
        For major domain mismatches (e.g., chef resume for data science role), relevance_score should be very low (0.0-0.2).
        """
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            return result.get('relevance_score', 0.0), result.get('explanation', 'No explanation')
            
        except json.JSONDecodeError:
            return calculate_semantic_fallback(resume_text, jd_text)
            
    except Exception as e:
        logger.error(f"LLM semantic analysis failed: {e}")
        return calculate_semantic_fallback(resume_text, jd_text)

def calculate_semantic_fallback(resume_text, jd_text):
    """Fallback semantic calculation using sentence transformers."""
    if not model:
        return 0.0, "Model not available"
    
    try:
        # Simple preprocessing
        resume_clean = ' '.join(resume_text.split()[:200])  # First 200 words
        jd_clean = ' '.join(jd_text.split()[:200])
        
        embeddings = model.encode([resume_clean, jd_clean])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return max(0.0, min(1.0, similarity)), "Embedding similarity"
        
    except Exception as e:
        logger.error(f"Semantic fallback failed: {e}")
        return 0.0, "Calculation failed"

# --- 7. ENHANCED SCORING WITH DOMAIN MISMATCH DETECTION ---
def calculate_comprehensive_score(resume_skills, resume_category, jd_info, semantic_score):
    """Calculate comprehensive score with domain mismatch detection."""
    required_skills = set(jd_info.get('must_have_skills', []))
    job_category = jd_info.get('role_category', 'general')
    
    # Check for major domain mismatches
    domain_mismatch = False
    if job_category == 'data science' or job_category == 'data analyst':
        if resume_category == 'culinary' or resume_category == 'sales':
            domain_mismatch = True
    elif job_category == 'culinary':
        if resume_category == 'technical':
            domain_mismatch = True
    
    # Calculate skill match score
    if required_skills:
        matched_skills = resume_skills.intersection(required_skills)
        skill_score = len(matched_skills) / len(required_skills)
    else:
        skill_score = 0.0
    
    # Apply domain mismatch penalty
    if domain_mismatch:
        final_score = min(0.2, skill_score * 0.3)  # Severely penalize domain mismatches
    else:
        # Normal scoring for domain-appropriate resumes
        final_score = (skill_score * 0.7) + (semantic_score * 0.3)
    
    # Determine verdict
    if domain_mismatch:
        verdict = "Rejected - Domain Mismatch"
    elif final_score >= 0.8:
        verdict = "High"
    elif final_score >= 0.6:
        verdict = "Medium"
    elif final_score >= 0.4:
        verdict = "Low"
    else:
        verdict = "Rejected"
    
    return final_score, verdict, domain_mismatch

# --- 8. ENHANCED FEEDBACK GENERATION ---
async def generate_comprehensive_feedback(resume_skills, jd_info, semantic_score, domain_mismatch):
    """Generate comprehensive feedback using LLM."""
    if not gemini_api_configured:
        return generate_feedback_fallback(resume_skills, jd_info, semantic_score, domain_mismatch)
    
    try:
        missing_skills = set(jd_info.get('must_have_skills', [])) - resume_skills
        
        prompt = f"""
        Generate constructive feedback for a job applicant:
        
        Job Role: {jd_info.get('job_title', 'Unknown')}
        Required Skills: {', '.join(jd_info.get('must_have_skills', []))}
        Candidate Skills: {', '.join(resume_skills) if resume_skills else 'Limited skills detected'}
        Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}
        Domain Mismatch: {'Yes' if domain_mismatch else 'No'}
        
        Provide specific, actionable feedback in 2-3 sentences. Focus on:
        1. If domain mismatch: explain why this role isn't suitable
        2. If skills missing: specific learning recommendations
        3. How to improve resume presentation
        
        Be direct but constructive.
        """
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Feedback generation failed: {e}")
        return generate_feedback_fallback(resume_skills, jd_info, semantic_score, domain_mismatch)

def generate_feedback_fallback(resume_skills, jd_info, semantic_score, domain_mismatch):
    """Fallback feedback generation."""
    if domain_mismatch:
        return f"This resume appears to be for a different field than the {jd_info.get('job_title', 'target role')}. Consider applying for positions that better match your background and experience."
    
    missing_skills = set(jd_info.get('must_have_skills', [])) - resume_skills
    if missing_skills:
        return f"Consider developing these key skills: {', '.join(list(missing_skills)[:3])}. Add relevant projects or certifications to strengthen your profile."
    
    return "Your profile shows good alignment with the role requirements. Consider highlighting specific achievements and quantifiable results to strengthen your application."

# --- 9. MAIN ANALYSIS PIPELINE ---
async def run_comprehensive_analysis(resume_filepath, jd_text):
    """Main analysis pipeline with LLM integration."""
    try:
        # Parse resume
        resume_text = parse_file(resume_filepath)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not parse resume file")

        # Analyze JD with LLM
        jd_info = await analyze_jd_with_llm(jd_text)
        
        # Extract resume skills and determine category
        resume_skills, resume_category = extract_resume_skills_comprehensive(resume_text)
        
        # Calculate semantic relevance with LLM
        semantic_score, semantic_explanation = await calculate_semantic_relevance_with_llm(
            resume_text, jd_text, jd_info
        )
        
        # Calculate comprehensive score
        final_score, verdict, domain_mismatch = calculate_comprehensive_score(
            resume_skills, resume_category, jd_info, semantic_score
        )
        
        # Generate feedback
        feedback = await generate_comprehensive_feedback(
            resume_skills, jd_info, semantic_score, domain_mismatch
        )
        
        # Calculate missing elements
        missing_skills = list(set(jd_info.get('must_have_skills', [])) - resume_skills)
        
        return {
            "final_score": round(final_score, 2),
            "verdict": verdict,
            "feedback": feedback,
            "missing_elements": {
                "missing_skills": missing_skills,
                "domain_mismatch": domain_mismatch
            },
            "detailed_analysis": {
                "job_title": jd_info.get('job_title', 'Unknown'),
                "job_category": jd_info.get('role_category', 'general'),
                "resume_category": resume_category,
                "skills_found": list(resume_skills),
                "semantic_score": round(semantic_score, 2),
                "semantic_explanation": semantic_explanation
            }
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- 10. FASTAPI APPLICATION ---
def create_app():
    app = FastAPI(
        title="Resume Relevance Analyzer API",
        description="AI-powered resume evaluation system for job matching",
        version="2.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Resume Analyzer API is running"}
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "llm_configured": gemini_api_configured
        }
    
    @app.post("/analyze")
    async def analyze_resume(
        resume: UploadFile = File(..., description="Resume file (PDF or DOCX)"),
        jd_text: str = Form(..., description="Job description text")
    ):
        """Analyze resume relevance against job description."""
        
        # Validate file type
        if not resume.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Save uploaded file temporarily
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=resume.filename) as tmp_file:
                content = await resume.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Run comprehensive analysis
            result = await run_comprehensive_analysis(tmp_file_path, jd_text)
            
            # Add resume filename to result
            result["resume_filename"] = resume.filename
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    @app.post("/batch-analyze")
    async def batch_analyze_resumes(
        resumes: List[UploadFile] = File(..., description="Multiple resume files"),
        jd_text: str = Form(..., description="Job description text")
    ):
        """Batch analyze multiple resumes."""
        
        results = []
        
        for resume in resumes:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=resume.filename) as tmp_file:
                    content = await resume.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                result = await run_comprehensive_analysis(tmp_file_path, jd_text)
                result["resume_filename"] = resume.filename
                results.append(result)
                
            except Exception as e:
                results.append({
                    "resume_filename": resume.filename,
                    "error": str(e),
                    "final_score": 0,
                    "verdict": "Error"
                })
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return JSONResponse(content={
            "total_resumes": len(results),
            "successful_analyses": len([r for r in results if 'error' not in r]),
            "failed_analyses": len([r for r in results if 'error' in r]),
            "results": results
        })
    
    return app

# --- 11. APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    print("Starting Enhanced Resume Analyzer API...")
    print(f"Model loaded: {model is not None}")
    print(f"Gemini API configured: {gemini_api_configured}")
    
    app = create_app()
    # Use port 7860 for HuggingFace Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)