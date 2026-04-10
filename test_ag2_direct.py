"""Test ag2's OpenAI wrapper directly with the planner config."""
import sys
sys.path.insert(0, ".")

from config.settings import get_settings
from config.model_profiles import get_planner_config

settings = get_settings()
config = get_planner_config(settings)

print("Config:", config)
print()

# Use ag2's own OpenAI wrapper (same path as ConversableAgent)
from autogen.oai import OpenAIWrapper

client = OpenAIWrapper(config_list=config["config_list"])

response = client.create(
    messages=[{"role": "user", "content": "Say hello in one word"}],
    model=settings.planner_model,
    max_tokens=10,
    temperature=0.6,
)

print("Response:", OpenAIWrapper.extract_text_or_completion_object(response))
