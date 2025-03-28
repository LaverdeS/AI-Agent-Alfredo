import os
import base64
import math
import pytz
import torch
import yaml
import pycountry
import subprocess
import sys
import numpy as np
import sounddevice as sd

from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.translation import TranslationTool
from tools.best_model_for_task import HFModelDownloadsTool
from tools.rag_transformers import retriever_tool

from transformers import pipeline
from Gradio_UI import GradioUI
from Gradio_UI_with_image import GradioUIImage
from dotenv import load_dotenv
from datetime import datetime
from skimage import io
from PIL import Image
from typing import Optional, Tuple
from IPython.display import Audio, display

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from io import BytesIO
from time import sleep

from smolagents.agents import ActionStep
from smolagents.cli import load_model
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    HfApiModel,
    TransformersModel,
    OpenAIServerModel,
    load_tool,
    Tool,
    tool,
    ToolCollection,
    E2BExecutor
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


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,  # Average speed for cargo planes
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)


@tool
def browser_automation(original_user_query:str)->str:
    """
    Browser automation is like “simulating a real user” and works for interactive,
    dynamic sites and when visual navigation is required to show the process to the user.
    Navigates the web using helium to answer a user query by appending helium_instructions to the original query
    by searching for text matches through the navigation.
    Args:
        original_user_query: The original
    """
    # Use sys.executable to ensure the same Python interpreter is used.
    result = subprocess.run(
        [sys.executable, "vision_web_browser.py", original_user_query],
        capture_output=True,  # Captures both stdout and stderr
        text=True  # Returns output as a string instead of bytes
    )
    print("vision_web_browser.py: ", result.stderr)
    return result.stdout


text_to_speech_pipe = pipeline(
    task="text-to-speech",
    model="suno/bark-small",
    device = 0 if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    )
text_to_speech_pipe.model.enable_cpu_offload()
text_to_speech_pipe.model.use_flash_attention_2=True

def speech_to_text(final_answer_text, agent_memory):
    text = f"[clears throat] {final_answer_text}"
    output = text_to_speech_pipe(text)
    # display(Audio(output["audio"], rate=output["sampling_rate"]))  # notebook
    audio = np.array(output["audio"], dtype=np.float32)
    print("Original audio shape:", audio.shape)

    # Adjust audio shape if necessary:
    if audio.ndim == 1:
        # Mono audio, should be fine. You can check if your device expects stereo.
        print("Mono audio... should be fine. You can check if your device expects stereo.")
    elif audio.ndim == 2:
        # Check if the number of channels is acceptable (e.g., 1 or 2)
        channels = audio.shape[1]
        if channels not in [1, 2]:
            # Try to squeeze extra dimensions
            audio = np.squeeze(audio)
            print("Squeezed audio shape:", audio.shape)
    else:
        # If audio has more dimensions than expected, flatten or reshape as needed
        audio = np.squeeze(audio)
        print("Squeezed audio shape:", audio.shape)

    # Play the audio using sounddevice
    try:
        sd.play(audio, output["sampling_rate"])
        sd.wait()  # Wait until audio playback is complete
    except Exception as e:
        print(f"Error playing audio: {e}")

    return True


def initialize_langfuse_opentelemetry_instrumentation():
    LANGFUSE_PUBLIC_KEY=os.environ.get("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY=os.environ.get("LANGFUSE_SECRET_KEY")
    LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
    
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


# telemetry
initialize_langfuse_opentelemetry_instrumentation()

# load tools from /tools/
final_answer = FinalAnswerTool()
visit_webpage = VisitWebpageTool()
translation = TranslationTool()
best_model_for_task = HFModelDownloadsTool()
transformers_retriever = retriever_tool

# load tools from smoloagents library
google_web_search = GoogleSearchTool()  # provider="serper" (SERPER_API_KEY) or "serpapi" (default)
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


# ceo_model = load_model("LiteLLMModel", "gpt-4o")   # or anthropic/claude-3-sonnet


ceo_model = HfApiModel(
max_tokens=2096,  # 8096 for manager
temperature=0.5,
model_id=  'https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',  # "meta-llama/Llama-3.3-70B-Instruct",  # 'https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',  # same as Qwen/Qwen2.5-Coder-32B-Instruct
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
        browser_automation,
        get_current_time_in_timezone,
        advanced_image_generation,
        image_generation_tool,
        transformers_retriever,
        language_detection,
        translation,
        calculate_cargo_travel_time
    ]

agent = CodeAgent(
    model=ceo_model,
    tools=tools,
    max_steps=20,  # 15 is good for a light manager, too much when there is no need of a manager
    verbosity_level=2,
    grammar=None,
    # planning_interval=5,  # (add more steps for heavier reasoning, leave default if not manager)  # test for crashing issues.
    name="Alfredo",
    description="CEO",
    prompt_templates=prompt_templates,
    # executor_type="e2b",  # security, could also be "docker" (set keys)
    # sandbox=E2BSandbox()  (or E2BExecutor?),
    # step_callbacks=[save_screenshot],  # todo: configure the web_navigation agent as a separate agent and manage it with alfred
    final_answer_checks=[speech_to_text],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
        "requests",
        "helium",
    ],
    # I could also add the authorized_imports from a LIST_SAFE_MODULES
)

agent.python_executor("from helium import *")   # agent.state

# agent.push_to_hub('laverdes/Alfredo')
agent.visualize()

# prompt = ("navigate to a random wikipedia page and give me a summary of the content, then make a single image representing all the content")
# agent.run(prompt)

GradioUI(agent).launch()
#GradioUIImage(agent).launch()