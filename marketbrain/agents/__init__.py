import os 
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1,
               api_key=os.getenv("GROQ_API_KEY"), max_tokens = 500)

def state_parser(content: str) -> dict:

    content = content.strip()
    if content.startswith("'''"):
        for part in content.split("'''"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break
    return json.loads(content.strip())



