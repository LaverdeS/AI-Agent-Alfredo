from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    HfApiModel,
    load_tool,
    tool
)
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.translation import TranslationTool

from Gradio_UI import GradioUI


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
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
def conversational_utterance(user_content:str)-> str:
    """A tool that replies to a single casual query or message that does not require any other tool to be triggered
    Args:
        user_content: the message or query such as 'Hi!', 'How are you?', 'What are you?', 'tell me a joke'
    """
    messages = [
      {"role": "user", "content": [{"type": "text", "text": user_content}]}
    ]
    return model(messages).content


final_answer = FinalAnswerTool()
prefered_web_search = GoogleSearchTool()
alternative_web_search = DuckDuckGoSearchTool()
visit_webpage = VisitWebpageTool()
translation_tool = TranslationTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
language_detection_tool = load_tool("team-language-detector/LanguageDetector", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        final_answer, 
        prefered_web_search, 
        alternative_web_search,
        visit_webpage, 
        get_current_time_in_timezone, 
        conversational_utterance, 
        image_generation_tool,
        language_detection_tool,
        translation_tool
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()