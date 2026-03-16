from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from medicalrag import rag_agent   # ← your exact filename

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "online"}

@app.post("/chat")
async def chat(req: ChatRequest):
    result = rag_agent.invoke({
        "messages":      [HumanMessage(content=req.message)],
        "safety_status": "proceed",
        "safety_msg":    ""
    })
    return {"response": result["messages"][-1].content}
```
