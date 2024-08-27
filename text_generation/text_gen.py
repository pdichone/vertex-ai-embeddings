import warnings
import sys

sys.path.append("../")

from utils import plot_2D


# Ignore all warnings
warnings.filterwarnings("ignore")

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import os


load_dotenv()
key_path = "../vertex-ai-course.json"  # Path to the json key associated with your service account from google cloud

# Create credentials object

credentials = Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

if credentials.expired:
    credentials.refresh(Request())

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

import vertexai

# initialize vertex
vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

from vertexai.language_models import TextGenerationModel

model = TextGenerationModel.from_pretrained("text-bison@001")

prompt = "I'm an aspiring chef interested in molecular gastronomy. \
What are some beginner techniques I can experiment with at home?"
# print(model.predict(prompt=prompt).text)

# == Classification ==
prompt = """
I want you to choose the most appropriate option from the list below and then elaborate on why you selected that option.

Question: "What is the best way to manage a remote team?"

Options: 
1. Regular video meetings
2. Detailed documentation
3. Flexible working hours
4. Performance-based incentives

Please select one of the options and explain why it is the best choice.
"""
# print(model.predict(prompt=prompt).text)

# ==== Extract information from text into a table (or other structured format)
prompt = """The newly formed tech startup InnovateX is quickly making waves in the industry. 
The team is led by CEO James Thompson (Jessica Lee), a visionary with a knack for identifying market trends. 
Chief Technology Officer Alan Smith (David Chen) is known for his groundbreaking work in artificial intelligence, 
while the Chief Marketing Officer Sophia Green (Isabella Lopez) brings creative strategies to the table. 
Their efforts are supported by the Operations Manager, Liam Brown (Michael Johnson), 
who ensures that all projects run smoothly and efficiently. 
Additionally, they have recently hired a data scientist, Olivia White (Emily Davis), 
who is responsible for analyzing market data to drive decision-making.

Extract the team members, their roles from above message as a JSON format.
"""

# print(model.predict(prompt=prompt).text)

# == Adjust the temperature (randomness and creativity)==

temperature = 0.9  # 0.0 is deterministic, 1.0 is maximum randomness change this to see the results
prompt = "Finish the following sentence: \
Before starting the renovation, \
I opened my toolbox to grab the:"

response = model.predict(
    prompt=prompt,
    temperature=temperature,
)
# print(f"[temperature = {temperature}]")

# print(response.text)

# ====== Adjust the top-P and top-n parameters =====
top_p = 0.7
top_k = 20
prompt = "Create a marketing slogan for a line of jackets \
that features dancing penguins and pineapples."
response = model.predict(prompt=prompt, temperature=0.9, top_p=top_p, top_k=top_k)
# print(f"[top_p = {top_p}]")
# print(response.text)

# == Use case - Transcript summarization and extraction  ==
# JSON transcript
transcript = """
{
  "call_id": "12345",
  "timestamp": "2024-08-23T10:30:00Z",
  "agent": {
    "name": "John Doe",
    "employee_id": "A789",
    "department": "Technical Support"
  },
  "customer": {
    "name": "Jane Smith",
    "customer_id": "C456",
    "account_number": "9876543210"
  },
  "transcript": [
    {"speaker": "Agent", "timestamp": "2024-08-23T10:30:15Z", "text": "Hello, Jane. My name is John, and I'm with the Technical Support team. I understand you're having some issues with your network configuration. How can I assist you today?"},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:31:00Z", "text": "Hi, John. Yes, I'm experiencing a problem with my BGP peering. I'm getting continuous route flaps, and my convergence time is abnormally high. I tried adjusting the MED and tweaking the AS path prepend, but nothing seems to stabilize the routes."},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:32:30Z", "text": "I see. That sounds frustrating. Just to confirm, you're using BGP for your external routing, and the issue is occurring when you're trying to exchange routes with your upstream ISP?"},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:32:45Z", "text": "Correct. The instability started after we implemented a new route reflector. I'm seeing a lot of BGP churn, and the RIB is getting overwhelmed, leading to route dampening."},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:34:10Z", "text": "Understood. Let's troubleshoot this step by step. First, let's check your route reflector configuration. Often, issues like this arise from suboptimal iBGP topology. Are you using the ORR (Optimal Route Reflection) feature, and have you enabled the BGP Add-Paths capability on your route reflectors?"},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:35:00Z", "text": "We haven't implemented ORR yet, but BGP Add-Paths is enabled. However, the problem persists even with these optimizations."},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:36:30Z", "text": "Alright. Let's focus on the RIB-IN and RIB-OUT. Have you applied any inbound route filtering, such as prefix lists or route maps, that could be affecting the advertised paths? Also, check if the BGP scanner interval is set too aggressively—it could cause unnecessary churn."},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:38:00Z", "text": "I did configure a prefix list, but it's quite broad. The BGP scanner interval is set to the default value. Should I consider tuning it?"},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:39:15Z", "text": "Yes, I recommend adjusting the BGP scanner interval to reduce the frequency of route validation. Additionally, review your prefix list and consider tightening it to prevent unnecessary routes from entering your RIB. This will help in minimizing the route churn. Also, ensure that your route reflectors are not becoming a bottleneck—check CPU and memory utilization on those devices."},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:40:30Z", "text": "Got it. I'll make those adjustments. Should I also consider disabling route dampening temporarily to see if it stabilizes the network?"},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:41:45Z", "text": "That's a good idea. Disabling route dampening will allow you to isolate whether it's contributing to the instability. Once the network is stable, you can re-enable it with more conservative parameters. After making these changes, monitor your BGP sessions for any improvements in convergence time and stability."},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:43:00Z", "text": "Thank you, John. I'll implement these changes and keep an eye on the network. Hopefully, this will resolve the issue."},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:43:30Z", "text": "You're welcome, Jane. If you run into any more issues or need further assistance, don't hesitate to reach out. I'll keep your case open for a few days in case you need any follow-up support."},
    {"speaker": "Customer", "timestamp": "2024-08-23T10:44:00Z", "text": "I appreciate that. Thanks again for your help!"},
    {"speaker": "Agent", "timestamp": "2024-08-23T10:44:15Z", "text": "My pleasure. Have a great day!"}
  ]
}
"""

# Define the prompt to extract sentiment and next steps
prompt = f"""
Given the following transcript of a technical support call:

{transcript}

1. Determine the overall sentiment of the conversation.
2. Extract and list the next steps provided by the agent to the customer.
and these steps need to be in a bullet point format.
"""
# Generate a response using the Bison model
response = model.predict(prompt=prompt, max_output_tokens=200)


# Print the response
print(response.text)
