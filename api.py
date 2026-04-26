from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_setup import initialize_agent
import os
import logging
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Lazy initialization - agent will be created on first request
agent = None
tools_dict = None

def get_agent():
    """Lazy initialization of the agent"""
    global agent, tools_dict
    if agent is None:
        logger.info("Initializing MedRAX agent...")
        
        selected_tools = [
            "ImageVisualizerTool",
            "DicomProcessorTool",
            "ChestXRayClassifierTool",
            "ChestXRaySegmentationTool",
            "ChestXRayReportGeneratorTool",
            "XRayVQATool",
        ]

        openai_kwargs = {}
        if os.getenv("OPENAI_API_KEY"):
            openai_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")

        try:
            agent, tools_dict = initialize_agent(
                "medrax/docs/system_prompts.txt",
                tools_to_use=selected_tools,
                model_dir=r"C:\model-weights",
                temp_dir="temp",
                device="cpu",
                model="gpt-4o",
                openai_kwargs=openai_kwargs
            )
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")
    
    return agent

class Query(BaseModel):
    input: str

@app.post("/query")
def query(q: Query):
    try:
        agent = get_agent()
        
        result = agent.workflow.invoke(
            {
                "messages": [
                    HumanMessage(content=q.input)
                ]
            },
            config={
                "configurable": {
                    "thread_id": "default_thread"
                }
            }
        )

        return {
            "output": result["messages"][-1].content
        }
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}