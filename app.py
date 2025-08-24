import gradio as gr

with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("# Inference Provider")
        gr.Markdown("This Space showcases the Qwen/Qwen3-Coder-480B-A35B-Instruct model, served by the hyperbolic API. Sign in with your Hugging Face account to use this API.")
        button = gr.LoginButton("Sign in")
    gr.load("models/Qwen/Qwen3-Coder-480B-A35B-Instruct", accept_token=button, provider="hyperbolic")
    
demo.launch()