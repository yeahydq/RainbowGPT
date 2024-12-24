import gradio as gr
from Rainbow_utils.model_config_manager import ModelConfigManager

class RainbowModelManager:
    def __init__(self):
        self.model_manager = ModelConfigManager()
        # 预定义模型列表
        self.gpt_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "Custom"]
        self.private_models = ["llama2", "mistral", "qwen", "Custom"]
        self.create_interface()
    
    def update_model_config(self, model_type, model_select, custom_model_name, api_base, api_key, temperature):
        try:
            # 确定最终使用的模型名称
            final_model_name = custom_model_name if model_select == "Custom" and custom_model_name.strip() else model_select
            
            if model_type == "GPT Model":
                if api_key.strip():  # 如果提供了新的API key
                    self.model_manager.gpt_config.api_key = api_key
                self.model_manager.set_gpt_config(final_model_name, temperature)
                self.model_manager.use_gpt_model()
            else:
                # 对于私有模型，总是使用新提供的配置
                self.model_manager.set_private_llm_config(
                    final_model_name,
                    api_base if api_base.strip() else "http://localhost:8000",
                    api_key if api_key.strip() else "",
                    temperature
                )
                self.model_manager.use_private_llm_model()
            
            active_config = self.model_manager.get_active_config()
            return (f"Successfully updated configuration:\n"
                    f"Model Type: {model_type}\n"
                    f"Model Name: {active_config.model_name}\n"
                    f"API Base: {active_config.api_base}\n"
                    f"API Key: {active_config.get_masked_key()}\n"
                    f"Temperature: {active_config.temperature}")
        except Exception as e:
            return f"Error updating configuration: {str(e)}"
    
    def update_model_visibility(self, model_type, custom_model_box, model_select, api_base, api_key):
        """更新界面组件的可见性和值"""
        try:
            if model_type == "GPT Model":
                return [
                    # custom_model_box (Textbox)
                    gr.update(visible=False, value=""),
                    # model_select (Dropdown)
                    gr.update(choices=self.gpt_models, value=self.gpt_models[0]),
                    # api_base (Textbox)
                    gr.update(value="https://api.chatanywhere.tech"),
                    # api_key (Textbox)
                    gr.update(value="")
                ]
            else:
                return [
                    # custom_model_box (Textbox)
                    gr.update(visible=False, value=""),
                    # model_select (Dropdown)
                    gr.update(choices=self.private_models, value=self.private_models[0]),
                    # api_base (Textbox)
                    gr.update(value="http://localhost:8000"),
                    # api_key (Textbox)
                    gr.update(value="")
                ]
        except Exception as e:
            print(f"Error in update_model_visibility: {str(e)}")
            return [gr.update() for _ in range(4)]
    
    def update_custom_visibility(self, model_name):
        """更新自定义模型输入框的可见性和值"""
        if model_name == "Custom":
            return gr.update(visible=True, value="")  # 当选择Custom时显示空输入框
        return gr.update(visible=False, value=model_name)  # 否则隐藏并保存当前选择的模型名称
    
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
            gr.Markdown("## RainbowGPT Model Configuration")
            
            with gr.Row():
                with gr.Column(scale=2):
                    model_type = gr.Radio(
                        choices=["GPT Model", "Private LLM Model"],
                        label="Model Type",
                        value="GPT Model",
                        info="选择模型类型"
                    )
                    
                    with gr.Group():
                        model_select = gr.Dropdown(
                            choices=self.gpt_models,
                            label="Select Model",
                            value=self.gpt_models[0],
                            info="选择预定义模型或自定义"
                        )
                        
                        custom_model_box = gr.Textbox(
                            label="Custom Model Name",
                            placeholder="输入自定义模型名称",
                            visible=False,
                            info="当选择Custom时可用",
                            value=""  # 确保初始值为空
                        )
                        
                        api_base = gr.Textbox(
                            value="https://api.chatanywhere.tech",
                            label="API Base URL",
                            info="API基础URL"
                        )
                        
                        api_key = gr.Textbox(
                            value="",
                            label="API Key",
                            type="password",
                            info="API密钥 (为安全起见不显示当前值)",
                            placeholder="输入新的 API Key"
                        )
                        
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=0,
                            step=0.1,
                            label="Temperature",
                            info="控制输出的随机性 (0-2)"
                        )
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(
                        label="Configuration Status",
                        lines=5,
                        info="显示当前配置状态"
                    )
            
            # 添加事件处理
            model_type.change(
                fn=self.update_model_visibility,
                inputs=[model_type, custom_model_box, model_select, api_base, api_key],
                outputs=[custom_model_box, model_select, api_base, api_key]
            )
            
            model_select.change(
                fn=self.update_custom_visibility,
                inputs=[model_select],
                outputs=[custom_model_box]
            )
            
            # 保存配置按钮
            save_btn = gr.Button("Save Configuration", variant="primary")
            save_btn.click(
                fn=self.update_model_config,
                inputs=[
                    model_type,
                    model_select,
                    custom_model_box,
                    api_base,
                    api_key,
                    temperature
                ],
                outputs=[status_output]
            )
            
            # 添加使用说明
            gr.Markdown("""
            ### 使用说明
            1. 选择模型类型（GPT Model 或 Private LLM Model）
            2. 从预定义列表选择模型或选择"Custom"输入自定义模型名称
            3. 根据需要修改API设置和温度参数
            4. 点击"Save Configuration"保存设置
            
            注意：
            - GPT Model默认使用OpenAI的API
            - Private LLM Model适用于自托管的模型
            - Temperature越高，输出越随机创造性；越低，输出越确定性
            """)
    
    def launch(self):
        return self.interface 