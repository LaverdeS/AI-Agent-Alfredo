from datetime import datetime
import pytz
import yaml
import pycountry

from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.translation import TranslationTool
from tools.best_model_for_task import HFModelDownloadsTool

from transformers import pipeline
from Gradio_UI import GradioUI

import os
import base64
from dotenv import load_dotenv

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from skimage import io
from PIL import Image

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    HfApiModel,
    TransformersModel,
    load_tool,
    Tool,
    tool,
    ToolCollection
)

# load .env vars
load_dotenv()


# fast prototyping tools
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone formatted as '%m/%d/%y %H:%M:%S'
    Args:
        timezone (str): A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.now(tz).strftime('%m/%d/%y %H:%M:%S')
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool
def language_detection(text:str)-> str:
    """Detects the language of the input text using basic xlm-roberta-base-language-detection.
     Args:
        text: the input message or wording to detect language from.
    """
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt)
    preds = pipe(text, return_all_scores=True, truncation=True, max_length=128)
    if preds:
        pred = preds[0]
        language_probabilities_dict = {p["label"]: float(p["score"]) for p in pred}
        predicted_language_code = max(language_probabilities_dict, key=language_probabilities_dict.get)
        tool_prediction_confidence = language_probabilities_dict[predicted_language_code]
        confidence_str = f"Tool Confidence: {tool_prediction_confidence}"
        predicted_language_code_str = f"Predicted language code (ISO 639): {predicted_language_code}/n{confidence_str}"
        try:
            predicted_language = pycountry.languages.get(alpha_2=predicted_language_code)
            if predicted_language:
                predicted_language_str = f"Predicted language: {predicted_language.name}/n{confidence_str}"
                return predicted_language_str 
            return predicted_language_code_str
            
        except Exception as e:
            return f"Error mapping country code to name (pycountry): {str(e)}/n{predicted_language_code_str}"
    else:
        return "None"


@tool
def advanced_image_generation(description:str)->Image.Image:
    """Generates an image using a textual description.
         Args:
            description: the textual description provided by the user to prompt a text-to-image model
        """
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate a detailed but short prompt (must be less than 900 characters) to generate an image based on the following description: {image_desc}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    image_url = DallEAPIWrapper().run(chain.run(description))
    image_array = io.imread(image_url)
    pil_image = Image.fromarray(image_array)
    return pil_image


# telemetry
def initialize_langfuse_opentelemetry_instrumentation():
    LANGFUSE_PUBLIC_KEY=os.environ.get("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY=os.environ.get("LANGFUSE_SECRET_KEY")
    LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
    
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

initialize_langfuse_opentelemetry_instrumentation()

# load tools from /tools/
final_answer = FinalAnswerTool()
visit_webpage = VisitWebpageTool()
translation = TranslationTool()
best_model_for_task = HFModelDownloadsTool()

# load tools from smoloagents library
google_web_search = GoogleSearchTool()
google_web_search.name = "google_web_search"
duckduckgo_web_search = DuckDuckGoSearchTool()
duckduckgo_web_search.name = "duckduckgo_web_search"

# load tools from hub and langchain
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)  # Tool.from_space("black-forest-labs/FLUX.1-schnell", name="image_generator", description="Generate an image from a prompt")
advanced_search_tool = Tool.from_langchain(load_tools(["searchapi"], allow_dangerous_tools=True)[0])  # serpapi is not real time scrapping
advanced_search_tool.name = "advanced_search_tool"

image_generation_tool_fast = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)


# alternative hf inference endpoint
model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',
custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

tools = [
        final_answer,
        best_model_for_task,
        advanced_search_tool,
        google_web_search,
        duckduckgo_web_search,
        visit_webpage, 
        get_current_time_in_timezone,
        advanced_image_generation,
        image_generation_tool,
        language_detection,
        translation
    ]

agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=10,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# agent.push_to_hub('laverdes/Alfredo')

GradioUI(agent).launch()