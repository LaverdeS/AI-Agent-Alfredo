import datetime
import requests
import pytz
import yaml
import pycountry

from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.translation import TranslationTool

from transformers import pipeline
from Gradio_UI import GradioUI
from typing import Optional

import os
import base64

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    HfApiModel,
    load_tool,
    tool
)


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone (str): A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def conversational_utterance(user_content: str, additional_context: Optional[str]="") -> str:
    """
    A tool that replies to a single casual query or message triggering any other tool is unfitted to reply.
    
    Args:
        user_content: A string with the user's message or query (e.g., "Hi!", "How are you?", "Tell me a joke").
        additional_context: An optional string with additional information (such as context, metadata, conversation history,
            or instructions) to be passed as an 'assistant' turn (a thought) in the conversation. 
    """
    system_context_message = f"""
        You are a highly intelligent, expert, and witty assistant who responds to user conversational messages.
        You function as a tool activated by user intention via AI agents. In addition to your native LLM capabilities,
        you have access to the following system tools that the user may leverage:
        {tools}
        You should mention these tools whenever relevant during the conversation.
    """
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_context_message}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"(additional_context: {additional_context})"}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]}
    ]
    return model(messages).content


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

# tools from /tools/
final_answer = FinalAnswerTool()
visit_webpage = VisitWebpageTool()
translation_tool = TranslationTool()

# tools from smoloagents library
prefered_web_search = GoogleSearchTool()
prefered_web_search.name = "preferred_web_search"
alternative_web_search = DuckDuckGoSearchTool()
alternative_web_search.name = "alternative_web_search"

# tools from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

tools = [
        final_answer, 
        prefered_web_search, 
        alternative_web_search,
        visit_webpage, 
        get_current_time_in_timezone, 
        conversational_utterance, 
        image_generation_tool,
        language_detection,
        translation_tool
    ]

agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=7,
    verbosity_level=2,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# agent.push_to_hub('laverdes/Alfredo')

GradioUI(agent).launch()