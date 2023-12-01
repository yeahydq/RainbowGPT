"""A simple web interactive chat demo based on gradio."""
import os
import dashscope
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = DASHSCOPE_API_KEY
# 打印 API 密钥
print(dashscope.api_key)


def slow_echo(message, history, llm_options_checkbox_group, temperature_num):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': message}]

    response = dashscope.Generation.call(
        model=str(llm_options_checkbox_group),
        messages=messages,
        result_format='message',  # set the result to be "message" format..
        temperature=float(temperature_num)  # Add temperature parameter
    )
    response = response["output"]["choices"][0]["message"]["content"]

    for i in range(0, len(response), int(10)):
        yield response[: i + int(10)]


with gr.Blocks() as RainbowGPT:
    with gr.Row():
        llm_options = ["qwen-turbo", "qwen-plus",
                       "qwen-max", ]
        llm_options_checkbox_group = gr.Dropdown(llm_options, label="LLM Model Select Options",
                                                 value=llm_options[2])

    temperature_num = gr.Slider(0, 1, render=False, label="Temperature")
    print_speed_step = gr.Slider(10, 20, render=False, label="Print Speed Step")

    custom_title = """
    <h1 style='text-align: center; margin-bottom: 1rem; font-family: "Courier New", monospace; 
               background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
               -webkit-background-clip: text;
               color: transparent;'>
        Qwen Web API Chat
    </h1>
   <p>通义千问超大规模语言模型</p>
   <p>qwen-turbo 8k tokens上下文，为了保障正常的使用和输出，API限定用户输入为6k tokens。</p>
   <p>qwen-plus 32k tokens上下文，为了保障正常的使用和输出，API限定用户输入为 30k tokens。</p>
   <p>qwen-max（限时免费开放中）30k tokens上下文，为了保障正常的使用和输出，API限定用户输入为 28k tokens。</p>
    """

    custom_description = """
    <div style='font-size: 12px; font-family: Arial, sans-serif; text-align: right; 
                background: linear-gradient(135deg, #ff4e50, #fc913a, #fed766, #4f98ca, #4f98ca, #fc913a, #ff4e50);
                -webkit-background-clip: text;
                color: transparent;'>
        <p>How to reach me: <a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></p>
    </div>
    """

    gr.ChatInterface(
        slow_echo, additional_inputs=[llm_options_checkbox_group, temperature_num],
        title=custom_title,
        description=custom_description,
        css=".gradio-container {background-color: #f0f0f0;}",  # Add your desired background color here
    )

RainbowGPT.queue().launch(share=True)
