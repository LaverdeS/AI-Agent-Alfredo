from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool

from Gradio_UI import GradioUI


@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

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
web_search = DuckDuckGoSearchTool()
visit_webpage = VisitWebpageTool()

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

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        final_answer, 
        web_search, 
        visit_webpage, 
        get_current_time_in_timezone, 
        conversational_utterance, 
        image_generation_tool
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

custom_css = """
/* Apply a nice background gradient to the whole page */
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
    font-family: 'Roboto', sans-serif;
}

/* Style the main container with a border, padding, and shadow */
.gradio-container {
    border: 2px solid #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
}

/* Example: style all headings inside the container */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    color: #ffffff;
}

/* Customize buttons appearance */
button {
    background-color: #4A90E2;
    border: none;
    border-radius: 5px;
    color: white;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
button:hover {
    background-color: #357ABD;
}
"""
GradioUI(agent).launch(css=custom_css)