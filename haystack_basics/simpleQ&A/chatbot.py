import random
import gradio as gr
from pipeline import AnswerGenerator


def response(query, history):
    generator = AnswerGenerator()
    answer = generator.get_answer(query)
    return str(answer)


demo = gr.ChatInterface(response, type="messages", autofocus=False)

if __name__ == "__main__":
    demo.launch()
