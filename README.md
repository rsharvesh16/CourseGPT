# CourseGPT

# CourseGPT - AI-Powered Course Authoring Platform

## 📌 Overview

CourseGPT is an innovative AI-powered platform that helps educators, trainers, and content creators design and develop high-quality courses with minimal effort. Leveraging large language models (LLMs) and modern web technologies, CourseGPT automates the creation of structured educational content while maintaining pedagogical soundness.

## ✨ Key Features

- **AI-Assisted Course Creation**: Generate complete course structures with modules and lessons using natural language prompts
- **Intelligent Content Generation**: Automatically create learning outcomes, key concepts, activities, and assessments
- **Structured Curriculum Design**: Maintain logical flow between modules and lessons
- **Collaborative Editing**: Edit and refine AI-generated content with a user-friendly interface
- **Pedagogically Sound**: Built with educational best practices in mind
- **Responsive Design**: Works on all devices

## 🛠️ Technology Stack

### Backend
- **Python** with **FastAPI** (Modern, fast web framework)
- **AWS Bedrock** (For accessing Mistral and other LLMs)
- **SQLite** (Lightweight database for development)
- **JWT** for authentication
- **LangChain** (For LLM orchestration)

### Frontend
- **Jinja2 Templates** (Server-side rendering)
- **HTML5**, **CSS3**, **JavaScript**
- **Bootstrap** (For responsive design)

### DevOps
- **Uvicorn** (ASGI server)
- **Environment variables** for configuration

## 🚀 Key Project Ideas Implemented

1. **AI-Powered Content Generation**
   - Dynamic prompt engineering for educational content
   - Structured JSON output parsing from LLM responses
   - Context-aware content regeneration for specific sections

2. **Curriculum Hierarchy Management**
   - Courses → Modules → Lessons structure
   - Prerequisite tracking between modules
   - Difficulty level and time estimation

3. **Educational Content Structuring**
   - Standardized lesson components (outcomes, concepts, activities, assessment)
   - Markdown support for rich content formatting
   - Activity and assessment generators

4. **User Experience**
   - Progressive enhancement approach
   - Responsive design for all devices
   - Contextual AI assistance throughout authoring process

5. **Security & Performance**
   - JWT authentication with secure cookies
   - Input validation and sanitization
   - Content size limits to prevent abuse


### Prerequisites
- Python 3.9+
- AWS account with Bedrock access
- Node.js (for optional frontend builds)

## 🌟 Future Enhancements

- Integration with LMS platforms (Moodle, Canvas)
- Multimedia content support (videos, interactive elements)
- Collaborative editing features
- Export to various formats (PDF, SCORM)
- Student progress tracking
- AI-powered content personalization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**CourseGPT** - Revolutionizing course creation with AI assistance! 🚀
