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
import json
from typing import List, Dict, Optional
from fastapi import HTTPException

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

# Setup SQLite database
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
    
    # Create courses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create modules table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS modules (
        id TEXT PRIMARY KEY,
        course_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        prerequisites TEXT,
        difficulty TEXT NOT NULL,
        estimated_time TEXT,
        FOREIGN KEY (course_id) REFERENCES courses (id)
    )
    ''')
    
    # Create lessons table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lessons (
        id TEXT PRIMARY KEY,
        module_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        learning_outcomes TEXT NOT NULL,  
        key_concepts TEXT NOT NULL,       
        content TEXT NOT NULL,
        activities TEXT NOT NULL,         
        assessment TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

# Initialize database when app starts
initialize_database()

# LLM Configuration
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN", None)  # Only needed for temporary credentials
)

# Get Mistral LLM
def get_llm():
    from langchain_aws import BedrockLLM
    return BedrockLLM(
        client=bedrock_client,
        model_id="mistral.mistral-large-2402-v1:0",
        model_kwargs={"temperature": 0.7, "top_p": 0.9, "max_tokens": 2000}
    )

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
    user_id: int
    title: str
    description: str
    created_at: str

class Module(BaseModel):
    id: str
    course_id: str
    title: str
    description: str
    prerequisites: List[str] = []
    difficulty: str
    estimated_time: str

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

# Database functions for courses, modules, and lessons
def save_course(user_id: int, title: str, description: str):
    course_id = str(uuid.uuid4())
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO courses (id, user_id, title, description) VALUES (?, ?, ?, ?)",
        (course_id, user_id, title, description)
    )
    
    conn.commit()
    conn.close()
    
    return course_id

def get_course(course_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM courses WHERE id = ?", (course_id,))
    course_data = cursor.fetchone()
    
    conn.close()
    
    if not course_data:
        return None
    
    return Course(
        id=course_data["id"],
        user_id=course_data["user_id"],
        title=course_data["title"],
        description=course_data["description"],
        created_at=course_data["created_at"]
    )

def get_user_courses(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM courses WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    courses_data = cursor.fetchall()
    
    conn.close()
    
    return [Course(
        id=course["id"],
        user_id=course["user_id"],
        title=course["title"],
        description=course["description"],
        created_at=course["created_at"]
    ) for course in courses_data]

def save_module(course_id: str, title: str, description: str, prerequisites: List[str], difficulty: str, estimated_time: str):
    module_id = str(uuid.uuid4())
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO modules (id, course_id, title, description, prerequisites, difficulty, estimated_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (module_id, course_id, title, description, json.dumps(prerequisites), difficulty, estimated_time)
    )
    
    conn.commit()
    conn.close()
    
    return module_id

def get_module(module_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM modules WHERE id = ?", (module_id,))
    module_data = cursor.fetchone()
    
    conn.close()
    
    if not module_data:
        return None
    
    return Module(
        id=module_data["id"],
        course_id=module_data["course_id"],
        title=module_data["title"],
        description=module_data["description"],
        prerequisites=json.loads(module_data["prerequisites"]) if module_data["prerequisites"] else [],
        difficulty=module_data["difficulty"],
        estimated_time=module_data["estimated_time"]
    )

def get_course_modules(course_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM modules WHERE course_id = ?", (course_id,))
    modules_data = cursor.fetchall()
    
    conn.close()
    
    return [Module(
        id=module["id"],
        course_id=module["course_id"],
        title=module["title"],
        description=module["description"],
        prerequisites=json.loads(module["prerequisites"]) if module["prerequisites"] else [],
        difficulty=module["difficulty"],
        estimated_time=module["estimated_time"]
    ) for module in modules_data]

def save_lesson(
    module_id: str,
    title: str,
    description: str,
    learning_outcomes: List[str],
    key_concepts: List[Dict[str, str]],
    content: str,
    activities: List[Dict[str, str]],
    assessment: List[Dict[str, str]],
) -> str:
    """Save a lesson to the database and return the lesson ID."""
    lesson_id = str(uuid.uuid4())
    conn = None
    try:
        # Convert all complex types to JSON strings
        learning_outcomes_json = json.dumps(learning_outcomes or [])
        key_concepts_json = json.dumps(key_concepts or [])
        activities_json = json.dumps(activities or [])
        assessment_json = json.dumps(assessment or [])
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO lessons (
                id, module_id, title, description, 
                learning_outcomes, key_concepts, 
                content, activities, assessment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson_id, module_id, title, description,
                learning_outcomes_json, key_concepts_json,
                content, activities_json, assessment_json
            )
        )
        
        conn.commit()
        return lesson_id
        
    except Exception as e:
        logger.error(f"Error saving lesson: {str(e)}")
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to save lesson to database"
        )
    finally:
        if conn:
            conn.close()

def update_lesson(lesson_id: str, title: str, description: str, learning_outcomes: List[str], 
                 key_concepts: List[Dict[str, str]], content: str, activities: List[Dict[str, str]], 
                 assessment: List[Dict[str, str]]):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Convert all lists and dictionaries to JSON strings
        learning_outcomes_json = json.dumps(learning_outcomes) if learning_outcomes else "[]"
        key_concepts_json = json.dumps(key_concepts) if key_concepts else "[]"
        activities_json = json.dumps(activities) if activities else "[]"
        assessment_json = json.dumps(assessment) if assessment else "[]"
        
        cursor.execute(
            "UPDATE lessons SET title = ?, description = ?, learning_outcomes = ?, key_concepts = ?, content = ?, activities = ?, assessment = ? WHERE id = ?",
            (
                title, 
                description, 
                learning_outcomes_json, 
                key_concepts_json, 
                content, 
                activities_json, 
                assessment_json, 
                lesson_id
            )
        )
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception(f"Error in update_lesson: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update lesson")
    finally:
        conn.close()

def get_module_lessons(module_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM lessons WHERE module_id = ?", (module_id,))
    lessons_data = cursor.fetchall()
    
    conn.close()
    
    lessons = []
    for lesson in lessons_data:
        try:
            lessons.append(Lesson(
                id=lesson["id"],
                module_id=lesson["module_id"],
                title=lesson["title"],
                description=lesson["description"],
                learning_outcomes=json.loads(lesson["learning_outcomes"]) if lesson["learning_outcomes"] else [],
                key_concepts=json.loads(lesson["key_concepts"]) if lesson["key_concepts"] else [],
                content=lesson["content"],
                activities=json.loads(lesson["activities"]) if lesson["activities"] else [],
                assessment=json.loads(lesson["assessment"]) if lesson["assessment"] else []
            ))
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON for lesson {lesson['id']}: {str(e)}")
            # Fallback to empty lists if JSON parsing fails
            lessons.append(Lesson(
                id=lesson["id"],
                module_id=lesson["module_id"],
                title=lesson["title"],
                description=lesson["description"],
                learning_outcomes=[],
                key_concepts=[],
                content=lesson["content"],
                activities=[],
                assessment=[]
            ))
    
    return lessons

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
    
    courses_list = get_user_courses(user.id)
    return templates.TemplateResponse("index.html", {"request": request, "user": user, "courses": courses_list})

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
    
    course_id = save_course(user.id, title, description)
    return RedirectResponse(url=f"/courses/{course_id}", status_code=303)

@app.get("/courses/{course_id}", response_class=HTMLResponse)
async def view_course(request: Request, course_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    course = get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    course_modules = get_course_modules(course_id)
    
    return templates.TemplateResponse(
        "course_detail.html", 
        {"request": request, "user": user, "course": course, "modules": course_modules}
    )

@app.get("/courses/{course_id}/modules/new", response_class=HTMLResponse)
async def new_module_form(request: Request, course_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    course = get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    return templates.TemplateResponse(
        "module_form.html", 
        {"request": request, "user": user, "course": course, "ai_generate": True}
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
    
    course = get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    prereq_list = [p.strip() for p in prerequisites.split(',') if p.strip()]
    
    module_id = save_module(course_id, title, description, prereq_list, difficulty, estimated_time)
    
    return RedirectResponse(url=f"/modules/{module_id}", status_code=303)

@app.post("/courses/{course_id}/modules/generate")
async def generate_module(
    request: Request,
    course_id: str,
    topic: str = Form(...),
):
    user = await get_current_user(request)
    if not user:
        return {"success": False, "error": "Authentication required"}
    
    course = get_course(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    try:
        llm = get_llm()
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["topic", "course_title"],
            template=MODULE_PROMPT
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate module content
        result = chain.invoke({"topic": topic, "course_title": course.title})
        
        try:
            # Clean up the result if it contains markdown code blocks
            cleaned_result = result['text']
            if '```json' in cleaned_result:
                cleaned_result = cleaned_result.split('```json')[1].split('```')[0].strip()
            elif '```' in cleaned_result:
                cleaned_result = cleaned_result.split('```')[1].split('```')[0].strip()
            
            module_data = json.loads(cleaned_result)
            
            # Normalize keys to handle various response formats
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
            
            # Create module
            module_id = save_module(course_id, title, description, prereq_list, difficulty, estimated_time)
            
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
    
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    course = get_course(module.course_id)
    module_lessons = get_module_lessons(module_id)
    
    return templates.TemplateResponse(
        "module_detail.html", 
        {"request": request, "user": user, "module": module, "course": course, "lessons": module_lessons}
    )

@app.get("/modules/{module_id}/lessons/new", response_class=HTMLResponse)
async def new_lesson_form(request: Request, module_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    course = get_course(module.course_id)
    
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
    
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
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
    
    lesson_id = save_lesson(
        module_id, title, description, outcomes_list, concepts_list, 
        content, activities_list, assessment_list
    )
    
    return RedirectResponse(url=f"/lessons/{lesson_id}", status_code=303)


@app.post("/modules/{module_id}/lessons/generate")
async def generate_lesson(
    request: Request,
    module_id: str,
    topic: str = Form(...),
):
    user = await get_current_user(request)
    if not user:
        return {"success": False, "error": "Authentication required"}
    
    module = get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    try:
        llm = get_llm()
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["topic", "module_title"],
            template=LESSON_PROMPT
        )
        
        # Use newer LangChain style (RunnableSequence instead of LLMChain)
        chain_result = (prompt | llm).invoke({"topic": topic, "module_title": module.title})
        
        try:
            # Clean up the result if it contains markdown code blocks
            cleaned_result = chain_result if isinstance(chain_result, str) else chain_result.get('text', '')
            if '```json' in cleaned_result:
                cleaned_result = cleaned_result.split('```json')[1].split('```')[0].strip()
            elif '```' in cleaned_result:
                cleaned_result = cleaned_result.split('```')[1].split('```')[0].strip()
            
            lesson_data = json.loads(cleaned_result)
            
            # Safety check for data size
            if len(json.dumps(lesson_data)) > 1000000:  # 1MB limit
                return {"success": False, "error": "Generated content is too large"}
            
            # Normalize keys to handle various response formats
            title = lesson_data.get("Title") or lesson_data.get("title", "Untitled Lesson")
            description = lesson_data.get("Description") or lesson_data.get("description", "")
            content = lesson_data.get("Content") or lesson_data.get("content", "")
            
            # Parse learning outcomes with safety limits
            outcomes_data = lesson_data.get("Learning Outcomes") or lesson_data.get("learning_outcomes", [])
            outcomes_list = []
            
            if isinstance(outcomes_data, str):
                outcomes_list = [o.strip() for o in outcomes_data.split('\n') if o.strip()][:10]  # Limit to 10 outcomes
            elif isinstance(outcomes_data, list) and len(outcomes_data) < 100:  # Safety check
                outcomes_list = outcomes_data[:10]  # Limit to 10 outcomes
            
            # Get key concepts with safety limits
            concepts_data = lesson_data.get("Key Concepts") or lesson_data.get("key_concepts", [])
            concepts_list = []
            
            if isinstance(concepts_data, str):
                lines = concepts_data.split('\n')[:20]  # Limit to 20 lines
                for line in lines:
                    if ':' in line:
                        term, definition = line.split(':', 1)
                        concepts_list.append({"title": term.strip(), "description": definition.strip()})
            elif isinstance(concepts_data, list) and len(concepts_data) < 100:  # Safety check
                for concept in concepts_data[:20]:  # Limit to 20 concepts
                    if isinstance(concept, dict) and "title" in concept and "description" in concept:
                        concepts_list.append({"title": concept["title"], "description": concept["description"]})
            
            # Get activities with safety limits
            activities_data = lesson_data.get("Learning Activities") or lesson_data.get("learning_activities", [])
            activities_list = []
            
            if isinstance(activities_data, str):
                activity_entries = activities_data.split('\n\n')[:10]  # Limit to 10 entries
                for entry in activity_entries:
                    if ':' in entry:
                        title, description = entry.split(':', 1)
                        activities_list.append({"title": title.strip(), "description": description.strip()})
            elif isinstance(activities_data, list) and len(activities_data) < 100:  # Safety check
                for activity in activities_data[:10]:  # Limit to 10 activities
                    if isinstance(activity, dict) and "title" in activity and "description" in activity:
                        activities_list.append({"title": activity["title"], "description": activity["description"]})
            
            # Get assessment with safety limits
            assessment_data = lesson_data.get("Assessment") or lesson_data.get("assessment", [])
            assessment_list = []
            
            if isinstance(assessment_data, str):
                assessment_entries = assessment_data.split('\n\n')[:10]  # Limit to 10 entries
                for entry in assessment_entries:
                    if ':' in entry:
                        title, description = entry.split(':', 1)
                        assessment_list.append({"title": title.strip(), "description": description.strip()})
            elif isinstance(assessment_data, list) and len(assessment_data) < 100:  # Safety check
                for assessment in assessment_data[:10]:  # Limit to 10 assessments
                    if isinstance(assessment, dict) and "title" in assessment and "description" in assessment:
                        assessment_list.append({"title": assessment["title"], "description": assessment["description"]})
            
            # Limit the content size to prevent memory issues
            if content and len(content) > 50000:  # 50KB limit
                content = content[:50000] + "...\n\n[Content truncated due to size limits]"
            
            # Create lesson
            lesson_id = save_lesson(
                module_id, title, description, outcomes_list, concepts_list, 
                content, activities_list, assessment_list
            )
            
            return {"success": True, "lesson_id": lesson_id, "data": lesson_data}
        
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON response: {cleaned_result}")
            return {"success": False, "error": "Failed to parse AI response", "raw_response": cleaned_result[:1000]}  # Limit error response size
        
    except MemoryError:
        logger.exception("Memory error while generating lesson")
        return {"success": False, "error": "Out of memory while generating lesson. Try with a simpler topic."}
    except Exception as e:
        logger.exception("Error generating lesson")
        return {"success": False, "error": str(e)}

def get_lesson(lesson_id: str) -> Optional[Lesson]:
    """Retrieve a lesson from the database by ID."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,))
        lesson_data = cursor.fetchone()
        
        if not lesson_data:
            return None
            
        # Deserialize JSON fields
        def safe_json_loads(data):
            try:
                return json.loads(data) if data else []
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for lesson {lesson_id}")
                return []
        
        return Lesson(
            id=lesson_data["id"],
            module_id=lesson_data["module_id"],
            title=lesson_data["title"],
            description=lesson_data["description"],
            learning_outcomes=safe_json_loads(lesson_data["learning_outcomes"]),
            key_concepts=safe_json_loads(lesson_data["key_concepts"]),
            content=lesson_data["content"],
            activities=safe_json_loads(lesson_data["activities"]),
            assessment=safe_json_loads(lesson_data["assessment"])
        )
        
    except Exception as e:
        logger.error(f"Error retrieving lesson {lesson_id}: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()
    

@app.get("/lessons/{lesson_id}", response_class=HTMLResponse)
async def view_lesson(request: Request, lesson_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    lesson = get_lesson(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    module = get_module(lesson.module_id)
    course = get_course(module.course_id) if module else None
    
    return templates.TemplateResponse(
        "lesson_detail.html", 
        {"request": request, "user": user, "lesson": lesson, "module": module, "course": course}
    )

@app.get("/lessons/{lesson_id}/edit", response_class=HTMLResponse)
async def edit_lesson_form(request: Request, lesson_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    lesson = get_lesson(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    module = get_module(lesson.module_id)
    course = get_course(module.course_id) if module else None
    
    return templates.TemplateResponse(
        "lesson_edit.html", 
        {"request": request, "user": user, "lesson": lesson, "module": module, "course": course}
    )

@app.post("/lessons/{lesson_id}/edit")
async def update_lesson_handler(
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
    
    lesson = get_lesson(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
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
    update_lesson(
        lesson_id, title, description, outcomes_list, concepts_list, 
        content, activities_list, assessment_list
    )
    
    return RedirectResponse(url=f"/lessons/{lesson_id}", status_code=303)

@app.post("/lessons/{lesson_id}/regenerate/{section}")
async def regenerate_section(
    request: Request,
    lesson_id: str,
    section: str,
    content: str = Form(...),
):
    user = await get_current_user(request)
    if not user:
        return {"success": False, "error": "Authentication required"}
    
    lesson = get_lesson(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    module = get_module(lesson.module_id)
    
    try:
        llm = get_llm()
        
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
            
            # Get current lesson data
            current_lesson = get_lesson(lesson_id)
            
            # Update the appropriate section
            if section == "learning_outcomes":
                updated_outcomes = generated_content
                key_concepts = current_lesson.key_concepts
                activities = current_lesson.activities
                assessment = current_lesson.assessment
            elif section == "key_concepts":
                updated_outcomes = current_lesson.learning_outcomes
                key_concepts = generated_content
                activities = current_lesson.activities
                assessment = current_lesson.assessment
            elif section == "activities":
                updated_outcomes = current_lesson.learning_outcomes
                key_concepts = current_lesson.key_concepts
                activities = generated_content
                assessment = current_lesson.assessment
            elif section == "assessment":
                updated_outcomes = current_lesson.learning_outcomes
                key_concepts = current_lesson.key_concepts
                activities = current_lesson.activities
                assessment = generated_content
            elif section == "content":
                # For content, we're just updating the content field and keeping all other fields the same
                update_lesson(
                    lesson_id,
                    current_lesson.title,
                    current_lesson.description,
                    current_lesson.learning_outcomes,
                    current_lesson.key_concepts,
                    generated_content,
                    current_lesson.activities,
                    current_lesson.assessment
                )
                return {"success": True, "data": generated_content}
            
            # Update the lesson with the new section
            update_lesson(
                lesson_id,
                current_lesson.title,
                current_lesson.description,
                updated_outcomes,
                key_concepts,
                current_lesson.content,
                activities,
                assessment
            )
            
            return {"success": True, "data": generated_content}
        
        except json.JSONDecodeError:
            # If it's not valid JSON, just return the raw text for certain sections
            if section == "content":
                # Update just the content field
                update_lesson(
                    lesson_id,
                    lesson.title,
                    lesson.description,
                    lesson.learning_outcomes,
                    lesson.key_concepts,
                    result['text'],
                    lesson.activities,
                    lesson.assessment
                )
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