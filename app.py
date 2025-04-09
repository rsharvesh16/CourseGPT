from fastapi import FastAPI, Request, Form, Depends, HTTPException, Cookie, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import boto3
import json
import uuid
import logging
from datetime import datetime, timedelta
import httpx
import asyncio
import sqlite3
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import huggingface_hub
from fastapi import FastAPI, Request, Form, Depends, HTTPException, Cookie, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CourseGPT", description="AI-Powered Course Authoring Platform")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Setup SQLite database for users
DATABASE_PATH = "coursegpt.db"

def initialize_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        disabled BOOLEAN NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

# Initialize database when app starts
initialize_database()

# Session storage for demo purposes (in production use a proper database for these as well)
courses = {}
modules = {}
lessons = {}

# LLM Configuration
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN", None)  # Only needed for temporary credentials
)

# Configure different LLM models
def get_llm(model_type="nova"):
    if model_type == "nova":
        from langchain_aws import BedrockLLM
        return BedrockLLM(
            client=bedrock_client,
            model_id="amazon.nova-pro-v1:0",
            model_kwargs={"max_new_tokens": 1000}
        )
    elif model_type == "llama":
        from langchain_aws import BedrockLLM
        return BedrockLLM(
            client=bedrock_client,
            model_id="meta.llama3-70b-instruct-v1:0",
            model_kwargs={"temperature": 0.7, "top_p": 0.9, "max_gen_len": 2000}
        )
    elif model_type == "mistral":
        from langchain_aws import BedrockLLM
        return BedrockLLM(
            client=bedrock_client,
            model_id="mistral.mistral-large-2402-v1:0",
            model_kwargs={"temperature": 0.7, "top_p": 0.9, "max_tokens": 2000}
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Get embeddings
def get_embeddings():
    from langchain_aws import BedrockEmbeddings
    return BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

# Data models
class Course(BaseModel):
    id: str
    title: str
    description: str
    created_at: str
    modules: List[str] = []

class Module(BaseModel):
    id: str
    course_id: str
    title: str
    description: str
    prerequisites: List[str] = []
    difficulty: str
    estimated_time: str
    lessons: List[str] = []

class Lesson(BaseModel):
    id: str
    module_id: str
    title: str
    description: str
    learning_outcomes: List[str] = []
    key_concepts: List[Dict[str, str]] = []
    content: str
    activities: List[Dict[str, str]] = []
    assessment: List[Dict[str, str]] = []

# Add these after your existing data models
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Authentication models
class User(BaseModel):
    id: int
    username: str
    email: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Database functions for user management
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_username(username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user_data = cursor.fetchone()
    
    conn.close()
    
    if not user_data:
        return None
    
    return UserInDB(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        hashed_password=user_data["hashed_password"],
        disabled=bool(user_data["disabled"])
    )

def create_user(username: str, email: str, password: str):
    hashed_password = pwd_context.hash(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (username, email, hashed_password)
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.close()
        if "username" in str(e).lower():
            raise ValueError("Username already exists")
        elif "email" in str(e).lower():
            raise ValueError("Email already exists")
        else:
            raise ValueError("Error creating user")
    
    conn.close()
    return True

# Prompt templates for AI generation
LESSON_PROMPT = """
You are an expert course creator with deep knowledge of educational design principles.
Create a detailed lesson plan for a lesson on "{topic}" that will be part of the module "{module_title}".

The lesson should include:

1. Title: A compelling and descriptive title for this lesson.
2. Description: A concise overview of what this lesson covers (2-3 sentences).
3. Learning Outcomes: 3-5 specific, measurable outcomes that begin with action verbs (e.g., "Explain", "Apply", "Analyze").
4. Key Concepts: 5-7 essential terms or concepts covered in this lesson, each with a definition.
5. Content: The core instructional material, organized into coherent sections with clear headings.
6. Learning Activities: 2-3 engaging activities that help reinforce the concepts, including detailed instructions.
7. Assessment: 2-3 questions or tasks that assess understanding of the material.

Format your response as a JSON object with these sections as keys. For the key concepts, activities, and assessment, 
provide an array of objects with "title" and "description" properties.

Be specific, clear, and pedagogically sound in your approach.
"""

MODULE_PROMPT = """
You are an expert curriculum designer with extensive experience in educational content organization.
Create a structured module plan for a module on "{topic}" that will be part of the course "{course_title}".

The module should include:

1. Title: A clear and descriptive title for this module.
2. Description: A concise overview of what this module covers (3-4 sentences).
3. Prerequisites: What prior knowledge or skills should learners have before starting this module?
4. Difficulty Level: Beginner, Intermediate, or Advanced
5. Estimated Completion Time: How long should this module take to complete?
6. Suggested Lessons: 4-6 lesson titles that should be included in this module, in a logical sequence.

Format your response as a JSON object with these sections as keys. For the suggested lessons, provide an array of strings.

Ensure that the module has a coherent structure with clear progression from one lesson to the next.
"""

# Functions for authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = get_user_by_username(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Add these functions for authentication
async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        user = get_user_by_username(username)
        return user
    except JWTError:
        return None

# Routes
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember: bool = Form(False)
):
    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": "Incorrect username or password"}
        )
    
    access_token_expires = timedelta(days=30 if remember else 1)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token", 
        value=access_token, 
        httponly=True, 
        max_age=access_token_expires.total_seconds() if remember else None
    )
    
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "register.html", 
            {"request": request, "error": "Passwords do not match"}
        )
    
    try:
        create_user(username, email, password)
    except ValueError as e:
        return templates.TemplateResponse(
            "register.html", 
            {"request": request, "error": str(e)}
        )
    
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse("index.html", {"request": request, "user": user, "courses": list(courses.values())})

@app.get("/courses/new", response_class=HTMLResponse)
async def new_course_form(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse("course_form.html", {"request": request, "user": user})

@app.post("/courses/new")
async def create_course(
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    course_id = str(uuid.uuid4())
    courses[course_id] = Course(
        id=course_id,
        title=title,
        description=description,
        created_at=datetime.now().isoformat(),
        modules=[]
    )
    return RedirectResponse(url=f"/courses/{course_id}", status_code=303)

@app.get("/courses/{course_id}", response_class=HTMLResponse)
async def view_course(request: Request, course_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if course_id not in courses:
        raise HTTPException(status_code=404, detail="Course not found")
    
    course = courses[course_id]
    course_modules = [modules[m_id] for m_id in course.modules if m_id in modules]
    
    return templates.TemplateResponse(
        "course_detail.html", 
        {"request": request, "user": user, "course": course, "modules": course_modules}
    )

@app.get("/courses/{course_id}/modules/new", response_class=HTMLResponse)
async def new_module_form(request: Request, course_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if course_id not in courses:
        raise HTTPException(status_code=404, detail="Course not found")
    
    return templates.TemplateResponse(
        "module_form.html", 
        {"request": request, "user": user, "course": courses[course_id], "ai_generate": True}
    )

@app.post("/courses/{course_id}/modules/new")
async def create_module(
    request: Request,
    course_id: str,
    title: str = Form(...),
    description: str = Form(...),
    prerequisites: str = Form(""),
    difficulty: str = Form("Beginner"),
    estimated_time: str = Form(""),
    ai_generate: bool = Form(False),
):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if course_id not in courses:
        raise HTTPException(status_code=404, detail="Course not found")
    
    module_id = str(uuid.uuid4())
    
    prereq_list = [p.strip() for p in prerequisites.split(',') if p.strip()]
    
    modules[module_id] = Module(
        id=module_id,
        course_id=course_id,
        title=title,
        description=description,
        prerequisites=prereq_list,
        difficulty=difficulty,
        estimated_time=estimated_time,
        lessons=[]
    )
    
    # Add module to course
    courses[course_id].modules.append(module_id)
    
    return RedirectResponse(url=f"/modules/{module_id}", status_code=303)

@app.post("/courses/{course_id}/modules/generate")
async def generate_module(
    request: Request,
    course_id: str,
    topic: str = Form(...),
    model_type: str = Form("nova"),
):
    user = await get_current_user(request)
    if not user:
        return {"success": False, "error": "Authentication required"}
    
    if course_id not in courses:
        raise HTTPException(status_code=404, detail="Course not found")
    
    try:
        llm = get_llm(model_type)
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["topic", "course_title"],
            template=MODULE_PROMPT
        )
        
        # Create chain
        from langchain.chains import LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate module content
        result = chain.invoke({"topic": topic, "course_title": courses[course_id].title})
        
        try:
            # Clean up the result if it contains markdown code blocks
            cleaned_result = result['text']
            if '```json' in cleaned_result:
                cleaned_result = cleaned_result.split('```json')[1].split('```')[0].strip()
            elif '```' in cleaned_result:
                cleaned_result = cleaned_result.split('```')[1].split('```')[0].strip()
            
            module_data = json.loads(cleaned_result)
            
            # Create module
            module_id = str(uuid.uuid4())
            
            # Handle various possible key formats in the response
            title = module_data.get("Title") or module_data.get("title", "Untitled Module")
            description = module_data.get("Description") or module_data.get("description", "")
            difficulty = module_data.get("Difficulty Level") or module_data.get("difficulty_level") or module_data.get("difficulty", "Beginner")
            estimated_time = module_data.get("Estimated Completion Time") or module_data.get("estimated_completion_time") or module_data.get("estimated_time", "")
            
            # Handle prerequisites which might be in different formats
            prereq_data = module_data.get("Prerequisites") or module_data.get("prerequisites", [])
            prereq_list = []
            
            if isinstance(prereq_data, str):
                prereq_list = [p.strip() for p in prereq_data.split(',') if p.strip()]
            elif isinstance(prereq_data, list):
                prereq_list = prereq_data
            
            # Handle suggested lessons
            lessons_data = module_data.get("Suggested Lessons") or module_data.get("suggested_lessons", [])
            
            modules[module_id] = Module(
                id=module_id,
                course_id=course_id,
                title=title,
                description=description,
                prerequisites=prereq_list,
                difficulty=difficulty,
                estimated_time=estimated_time,
                lessons=[]
            )
            
            # Add module to course
            courses[course_id].modules.append(module_id)
            
            return {"success": True, "module_id": module_id, "data": module_data}
        
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON response: {result}")
            return {"success": False, "error": "Failed to parse AI response", "raw_response": result}
    
    except Exception as e:
        logger.exception("Error generating module")
        return {"success": False, "error": str(e)}

@app.get("/modules/{module_id}", response_class=HTMLResponse)
async def view_module(request: Request, module_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if module_id not in modules:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules[module_id]
    course = courses.get(module.course_id)
    module_lessons = [lessons[l_id] for l_id in module.lessons if l_id in lessons]
    
    return templates.TemplateResponse(
        "module_detail.html", 
        {"request": request, "user": user, "module": module, "course": course, "lessons": module_lessons}
    )

@app.get("/modules/{module_id}/lessons/new", response_class=HTMLResponse)
async def new_lesson_form(request: Request, module_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if module_id not in modules:
        raise HTTPException(status_code=404, detail="Module not found")
    
    module = modules[module_id]
    course = courses.get(module.course_id)
    
    return templates.TemplateResponse(
        "lesson_form.html", 
        {"request": request, "user": user, "module": module, "course": course, "ai_generate": True}
    )

@app.post("/modules/{module_id}/lessons/new")
async def create_lesson(
    request: Request,
    module_id: str,
    title: str = Form(...),
    description: str = Form(...),
    learning_outcomes: str = Form(""),
    content: str = Form(""),
    key_concepts: str = Form(""),
    activities: str = Form(""),
    assessment: str = Form(""),
):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if module_id not in modules:
        raise HTTPException(status_code=404, detail="Module not found")
    
    lesson_id = str(uuid.uuid4())
    
    # Parse learning outcomes
    outcomes_list = [o.strip() for o in learning_outcomes.split('\n') if o.strip()]
    
    # Parse key concepts (format: term:definition, one per line)
    concepts_list = []
    for line in key_concepts.split('\n'):
        if ':' in line:
            term, definition = line.split(':', 1)
            concepts_list.append({"title": term.strip(), "description": definition.strip()})
    
    # Parse activities (format: title:description, separated by double newlines)
    activities_list = []
    activity_entries = activities.split('\n\n')
    for entry in activity_entries:
        if ':' in entry:
            title, description = entry.split(':', 1)
            activities_list.append({"title": title.strip(), "description": description.strip()})
    
    # Parse assessment (format: title:description, separated by double newlines)
    assessment_list = []
    assessment_entries = assessment.split('\n\n')
    for entry in assessment_entries:
        if ':' in entry:
            title, description = entry.split(':', 1)
            assessment_list.append({"title": title.strip(), "description": description.strip()})
    
    lessons[lesson_id] = Lesson(
        id=lesson_id,
        module_id=module_id,
        title=title,
        description=description,
        learning_outcomes=outcomes_list,
        key_concepts=concepts_list,
        content=content,
        activities=activities_list,
        assessment=assessment_list
    )
    
    # Add lesson to module
    modules[module_id].lessons.append(lesson_id)
    
    return RedirectResponse(url=f"/lessons/{lesson_id}", status_code=303)

@app.post("/modules/{module_id}/lessons/generate")
async def generate_lesson(
    request: Request,
    module_id: str,
    topic: str = Form(...),
    model_type: str = Form("nova"),
):
    user = await get_current_user(request)
    if not user:
        return {"success": False, "error": "Authentication required"}
    
    if module_id not in modules:
        raise HTTPException(status_code=404, detail="Module not found")
    
    try:
        llm = get_llm(model_type)
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["topic", "module_title"],
            template=LESSON_PROMPT
        )
        
        # Create chain
        from langchain.chains import LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate lesson content
        result = chain.invoke({"topic": topic, "module_title": modules[module_id].title})
        
        try:
            # Clean up the result if it contains markdown code blocks
            cleaned_result = result['text']
            if '```json' in cleaned_result:
                cleaned_result = cleaned_result.split('```json')[1].split('```')[0].strip()
            elif '```' in cleaned_result:
                cleaned_result = cleaned_result.split('```')[1].split('```')[0].strip()
            
            lesson_data = json.loads(cleaned_result)
            
            # Create lesson
            lesson_id = str(uuid.uuid4())
            
            # Normalize keys to handle various response formats
            title = lesson_data.get("Title") or lesson_data.get("title", "Untitled Lesson")
            description = lesson_data.get("Description") or lesson_data.get("description", "")
            content = lesson_data.get("Content") or lesson_data.get("content", "")
            
            # Parse learning outcomes
            outcomes_data = lesson_data.get("Learning Outcomes") or lesson_data.get("learning_outcomes", [])
            outcomes_list = []
            
            if isinstance(outcomes_data, str):
                outcomes_list = [o.strip() for o in outcomes_data.split('\n') if o.strip()]
            elif isinstance(outcomes_data, list):
                outcomes_list = outcomes_data
            
            # Get key concepts
            concepts_data = lesson_data.get("Key Concepts") or lesson_data.get("key_concepts", [])
            concepts_list = []
            
            if isinstance(concepts_data, str):
                for line in concepts_data.split('\n'):
                    if ':' in line:
                        term, definition = line.split(':', 1)
                        concepts_list.append({"title": term.strip(), "description": definition.strip()})
            elif isinstance(concepts_data, list):
                concepts_list = concepts_data
            
            # Get activities
            activities_data = lesson_data.get("Learning Activities") or lesson_data.get("learning_activities", [])
            activities_list = []
            
            if isinstance(activities_data, str):
                activity_entries = activities_data.split('\n\n')
                for entry in activity_entries:
                    if ':' in entry:
                        title, description = entry.split(':', 1)
                        activities_list.append({"title": title.strip(), "description": description.strip()})
            elif isinstance(activities_data, list):
                activities_list = activities_data
            
            # Get assessment
            assessment_data = lesson_data.get("Assessment") or lesson_data.get("assessment", [])
            assessment_list = []
            
            if isinstance(assessment_data, str):
                assessment_entries = assessment_data.split('\n\n')
                for entry in assessment_entries:
                    if ':' in entry:
                        title, description = entry.split(':', 1)
                        assessment_list.append({"title": title.strip(), "description": description.strip()})
            elif isinstance(assessment_data, list):
                assessment_list = assessment_data
            
            lessons[lesson_id] = Lesson(
                id=lesson_id,
                module_id=module_id,
                title=title,
                description=description,
                learning_outcomes=outcomes_list,
                key_concepts=concepts_list,
                content=content,
                activities=activities_list,
                assessment=assessment_list
            )
            
            # Add lesson to module
            modules[module_id].lessons.append(lesson_id)
            
            return {"success": True, "lesson_id": lesson_id, "data": lesson_data}
        
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON response: {result}")
            return {"success": False, "error": "Failed to parse AI response", "raw_response": result}
    
    except Exception as e:
        logger.exception("Error generating lesson")
        return {"success": False, "error": str(e)}

@app.get("/lessons/{lesson_id}", response_class=HTMLResponse)
async def view_lesson(request: Request, lesson_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if lesson_id not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    lesson = lessons[lesson_id]
    module = modules.get(lesson.module_id)
    course = courses.get(module.course_id) if module else None
    
    return templates.TemplateResponse(
        "lesson_detail.html", 
        {"request": request, "user": user, "lesson": lesson, "module": module, "course": course}
    )

@app.get("/lessons/{lesson_id}/edit", response_class=HTMLResponse)
async def edit_lesson_form(request: Request, lesson_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if lesson_id not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    lesson = lessons[lesson_id]
    module = modules.get(lesson.module_id)
    course = courses.get(module.course_id) if module else None
    
    return templates.TemplateResponse(
        "lesson_edit.html", 
        {"request": request, "user": user, "lesson": lesson, "module": module, "course": course}
    )

@app.post("/lessons/{lesson_id}/edit")
async def update_lesson(
    request: Request,
    lesson_id: str,
    title: str = Form(...),
    description: str = Form(...),
    learning_outcomes: str = Form(""),
    content: str = Form(""),
    key_concepts: str = Form(""),
    activities: str = Form(""),
    assessment: str = Form(""),
):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if lesson_id not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    lesson = lessons[lesson_id]
    
    # Parse learning outcomes
    outcomes_list = [o.strip() for o in learning_outcomes.split('\n') if o.strip()]
    
    # Parse key concepts (format: term:definition, one per line)
    concepts_list = []
    for line in key_concepts.split('\n'):
        if ':' in line:
            term, definition = line.split(':', 1)
            concepts_list.append({"title": term.strip(), "description": definition.strip()})
    
    # Parse activities (format: title:description, separated by double newlines)
    activities_list = []
    activity_entries = activities.split('\n\n')
    for entry in activity_entries:
        if ':' in entry:
            title, description = entry.split(':', 1)
            activities_list.append({"title": title.strip(), "description": description.strip()})
    
    # Parse assessment (format: title:description, separated by double newlines)
    assessment_list = []
    assessment_entries = assessment.split('\n\n')
    for entry in assessment_entries:
        if ':' in entry:
            title, description = entry.split(':', 1)
            assessment_list.append({"title": title.strip(), "description": description.strip()})
    
    # Update lesson
    lesson.title = title
    lesson.description = description
    lesson.learning_outcomes = outcomes_list
    lesson.key_concepts = concepts_list
    lesson.content = content
    lesson.activities = activities_list
    lesson.assessment = assessment_list
    
    return RedirectResponse(url=f"/lessons/{lesson_id}", status_code=303)

@app.post("/lessons/{lesson_id}/regenerate/{section}")
async def regenerate_section(
    request: Request,
    lesson_id: str,
    section: str,
    content: str = Form(...),
    model_type: str = Form("nova"),
):
    user = await get_current_user(request)
    if not user:
        return {"success": False, "error": "Authentication required"}
    
    if lesson_id not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    lesson = lessons[lesson_id]
    module = modules.get(lesson.module_id)
    
    try:
        llm = get_llm(model_type)
        
        # Define prompts for different sections
        prompts = {
            "learning_outcomes": f"""
                Create 3-5 specific, measurable learning outcomes for a lesson on "{lesson.title}".
                Each outcome should begin with an action verb (e.g., "Explain", "Apply", "Analyze").
                Format your response as a JSON array of strings.
                
                Additional context:
                {content}
            """,
            "key_concepts": f"""
                Create 5-7 essential terms or concepts for a lesson on "{lesson.title}".
                For each term, provide a clear and concise definition.
                Format your response as a JSON array of objects, each with "title" and "description" properties.
                
                Additional context:
                {content}
            """,
            "content": f"""
                Create the core instructional content for a lesson on "{lesson.title}".
                The content should be organized into coherent sections with clear headings.
                Make sure to cover all essential information related to the topic.
                
                Here are the learning outcomes for this lesson:
                {', '.join(lesson.learning_outcomes)}
                
                Use markdown formatting for better readability.
                
                Additional context:
                {content}
            """,
            "activities": f"""
                Create 2-3 engaging learning activities that help reinforce the concepts in the lesson "{lesson.title}".
                Each activity should include a title and detailed instructions.
                Format your response as a JSON array of objects, each with "title" and "description" properties.
                
                Additional context:
                {content}
            """,
            "assessment": f"""
                Create 2-3 assessment questions or tasks that measure understanding of the lesson "{lesson.title}".
                Include a mix of question types (e.g., multiple choice, short answer, application).
                Format your response as a JSON array of objects, each with "title" and "description" properties.
                
                Additional context:
                {content}
            """
        }
        
        if section not in prompts:
            raise HTTPException(status_code=400, detail="Invalid section")
        
        # Create chain
        from langchain.chains import LLMChain
        chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=[], template=prompts[section]))
        
        # Generate content
        result = chain.invoke({})
        
        try:
            cleaned_result = result['text']
            if '```json' in cleaned_result:
                cleaned_result = cleaned_result.split('```json')[1].split('```')[0].strip()
            elif '```' in cleaned_result:
                cleaned_result = cleaned_result.split('```')[1].split('```')[0].strip()
                
            generated_content = json.loads(cleaned_result)
            
            if section == "learning_outcomes":
                lesson.learning_outcomes = generated_content
            elif section == "key_concepts":
                lesson.key_concepts = generated_content
            elif section == "content":
                lesson.content = generated_content
            elif section == "activities":
                lesson.activities = generated_content
            elif section == "assessment":
                lesson.assessment = generated_content
            
            return {"success": True, "data": generated_content}
        
        except json.JSONDecodeError:
            # If it's not valid JSON, just return the raw text for certain sections
            if section == "content":
                lesson.content = result['text']
                return {"success": True, "data": result['text']}
            else:
                return {"success": False, "error": "Failed to parse AI response", "raw_response": result['text']}
    
    except Exception as e:
        logger.exception(f"Error generating {section}")
        return {"success": False, "error": str(e)}
    
# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)