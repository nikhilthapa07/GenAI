import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(
    api_key=API_KEY,
)


def get_temperature(cityName):
    print(f"===========get temp for {cityName}")
    # fake api call
    return "23 degree celcius"


def get_time(cityName):
    print(f"===========get time for {cityName}")
    # fake api call
    return "12:30 AM"


system_prompt = """
You are an helpful AI assistant who specializes in helping user.
You answer to the user by strictly taking these steps: plan, action, observe, output. You will only take one step at a time and wait for next step. When answering you will follow the output schema below:
You have predefined set of tools you can use to answer the user.
If the user's query can not be answered by using the tools you will politely respond the user that you can not help them.
Refer to the examples on how you will respond to the user.

output json schema:
{{
    "step": "string",
    "content": "string",
    "function": "the name of the function if the step is action",
    "input": "the input parameter for the function"
}}

Rules:
1. you will strictly follow the steps.
2. you will only take one step at a time and wait for next step.
3. You will comply with the output json schema
4. If the tool is available, you will pass the right parameter and use the return value when responding to user .

Available tools:
- "get_temperature": Takes a city name argument and returns temperature of the city
- "get_time": Takes a city name as argument and returns time of the city


Examples:
Input: what is the current temperature in dubai
Output: {{ "step": "plan", "content": "The user is interested in current temperature of the city. So I will use the get_temperature tool" }}
Output: {{"step": "action", "function": "get_temperature", "input": "dubai"}}
Output: {{"step": "observe", "content": "22 degree celcius" }}
Output: {{"step": "output", "content": "The current temperature of dubai is 22 degree celcius" }}


Input: why is the sky blue
Output: {{ "step": "plan", "content": "The user is interested in knowing why sky color is blue. In the provided list of tools, I do not have any tool to get that information." }}
Output: {{ "step": "output", "content": "Sorry, but I can not help you answer this"}}
"""

query = ""

while query.strip() == "":
    print("How can I help you.")
    query = input("> ")

messages_list = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query},
]

while True:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages_list,  # pyright: ignore
        )

        # print("response_content =======", response.choices[0].message.content)
        response_content = response.choices[0].message.content

        parsed_response = json.loads(response_content)  # pyright: ignore
        step = parsed_response.get("step")
        content = parsed_response.get("content")
        function = parsed_response.get("function")
        tool_input = parsed_response.get("input")

        if step == "output":
            print(f"🤖: {content}")
            break
        
        if step == "action":
            print(f"Calling Tool 🔨: {function}({tool_input})")

            # Actually call the function and get the result
            if function == "get_temperature":
                tool_result = get_temperature(tool_input)
            elif function == "get_time":
                tool_result = get_time(tool_input)
            else:
                tool_result = f"Error: unknown tool '{function}'"

            print(f"Tool Result: {tool_result}")

            # Feed the result back so the model can observe it
            observe_message = json.dumps({"step": "observe", "content": tool_result})
            messages_list.append({"role": "user", "content": observe_message})

        else:
            print(f"{step} 🤔: {content}")
        messages_list.append(
            {"role": "assistant", "content": response_content}  # pyright: ignore
        )

    except Exception as err:
        print("error...", err)
        sys.exit(1)
