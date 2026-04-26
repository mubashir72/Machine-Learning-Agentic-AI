import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from agent_setup import initialize_agent
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


if __name__ == "__main__":
    print("Initializing MedRAX agent...")
    
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

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir=r"C:\model-weights",
        temp_dir="temp",
        device="cpu",
        model="gpt-4o",
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )

    print("Creating Gradio interface...")
    demo = create_demo(agent, tools_dict)
    print("Launching MedRAX UI on http://127.0.0.1:7860/")
    demo.launch(server_name="127.0.0.1", server_port=7860)