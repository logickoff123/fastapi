from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
#from passlib.context import CryptContext
#from openai import OpenAI
from pydantic import BaseModel
import os

# Настройка FastAPI
app = FastAPI()

#client = OpenAI(
#    api_key="api key",  # This is the default and can be omitted
#  )
# Настройка OpenAI
#openai.api_key = "api key"
#openai.base_url = "api key"


DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True)
    role = Column(String) #TODO: роли добавить 


class Subject(Base):
    __tablename__ = "subject"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)


class Theme(Base):
    __tablename__ = "theme"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    subject_id = Column(Integer, ForeignKey("subject.id", ondelete="CASCADE"))


class Problem(Base):
    __tablename__ = "problem"
    id = Column(Integer, primary_key=True, index=True)
    raw_data = Column(String, unique=True, index=True)
    correct_answer = Column(String)
    theme_id = Column(Integer, ForeignKey("theme.id", ondelete="CASCADE"))


class UserAnswers(Base):
    __tablename__ = "user_answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    problem_id = Column(Integer, ForeignKey("problem.id", ondelete="CASCADE"))
    is_correct = Column(Integer)

# Модель для запроса и ответа
class QuestionRequest(BaseModel):
    topic: str = "math"
    difficulty: str = "easy" # Уровень сложности: easy, medium, hard
    amount: int = 10

class BaseResponse(BaseModel):
    response: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    

class QuestionBatchResponse(BaseModel):
    batch: list

Base.metadata.create_all(bind=engine)

# HTTPBasic для аутентификации
security = HTTPBasic()


def get_user(username: str, db):
    return db.query(User).filter(User.username == username).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(db, username: str, password: str):
    user = get_user(username, db)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


@app.post("/register/")
async def register_user(username: str, password: str, email: str):
    """
    Регистрация нового пользователя.
    """
    db = SessionLocal()
    try:
        # Проверяем, существует ли пользователь
        if get_user(username, db):
            raise HTTPException(status_code=400, detail="Пользователь уже существует.")
        
        # Создаем нового пользователя
        hashed_password = pwd_context.hash(password)
        new_user = User(username=username, hashed_password=hashed_password, email=email)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"username": new_user.username, "message": "Пользователь успешно зарегистрирован."}
    finally:
        db.close()

@app.get("/protected/")
async def read_protected(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Пример защищенного маршрута.
    """
    db = SessionLocal()
    try:
        user = authenticate_user(db, credentials.username, credentials.password)
        if not user:
            raise HTTPException(status_code=401, detail="Неверные учетные данные.")
        return {"message": f"Добро пожаловать, {user.username}!"}
    finally:
        db.close()

# Маршрут для генерации математических вопросов
@app.post("/generate_question/", response_model=list)
async def generate_question(request: QuestionRequest):
    """
    Генерирует математический вопрос и ответ с использованием OpenAI API.
    """
    try:
        new_batch = []
        for i in range(request.amount):
            #TODO: добавь словарь/json с конкретными подсказками для определенных тем
            prompt = (
                f"Generate a {request.difficulty} math question and provide the solution. "
                f"The topic is {request.topic}. Use random numbers, not just 7 and 5"
                "Write \"Answer:\" before the solution and don't write \"Question:\" before the question"
            )

            # Запрос к OpenAI API
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-4o-mini",
            #    max_tokens=150
            )
            # Извлечение ответа
            content = response.choices[0].message.content.strip()

            # Разделение на вопрос и ответ
            if "\nAnswer:" in content:
                question, answer = content.split("\nAnswer:")
            else:
                raise HTTPException(status_code=500, detail="Не удалось извлечь ответ.")
            new_batch.append({"question": question.strip(), "answer": answer.strip()})
        return new_batch

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации вопроса: {e}")
