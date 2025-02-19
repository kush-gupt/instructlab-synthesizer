import os as os
import gradio as gr
from llama_cpp import Llama
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

# Citing for reference: 
#  - Instruction Synthesizer model, paper and demo: https://huggingface.co/instruction-pretrain/instruction-synthesizer
#  - Deepseek-Coder-V2-Lite-Instruct as a pair programmer: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
#  - Example for incorporating llama_cpp with Gradio: https://huggingface.co/spaces/SpacesExamples/llama-cpp-python-cuda-gradio/blob/main/app.py

# llama-cpp model loading
def load_model():
    model_path = os.environ.get("MODEL_PATH", "~/Downloads/instruction-synthesizer.Q6_K.gguf")
    n_ctx = int(os.environ.get("N_CTX", 2048))
    n_threads = int(os.environ.get("N_THREADS", 5))
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
    return llm

# Load the local instruction-synthesizer model
llm = load_model()

# Parse the predictions returned from get_instruction_response_pairs
def parse_pred(pred):
    """Extract the list of instruction-response pairs from the prediction"""
    QA_str_list = pred.split('</END>')
    if not pred.endswith('</END>'):
        QA_str_list = QA_str_list[:-1]

    QA_list = []
    raw_questions = []
    for QA_str in QA_str_list:
        try:
            assert len(QA_str.split('<ANS>')) == 2, f'invalid QA string: {QA_str}'
            Q_str, A_str = QA_str.split('<ANS>')
            Q_str, A_str = Q_str.strip(), A_str.strip()
            assert Q_str.startswith('<QUE>'), f'invalid question string: {Q_str} in QA_str: {QA_str}'
            assert len(A_str) > 0, f'invalid answer string in QA_str: {QA_str}'
            Q_str = Q_str.replace('<QUE>', '').strip()
            assert Q_str.lower() not in raw_questions, f'duplicate question: {Q_str}'
            QA_list.append({'Q': Q_str, 'A': A_str})
            raw_questions.append(Q_str.lower())
        except:
            pass

    return QA_list

# Generate the instruction response pairs
def get_instruction_response_pairs(context):
    '''Prompt the synthesizer to generate instruction-response pairs based on the given context'''
    prompt = f'<s> <CON> {context} </CON>\n\n'

    # Generate predictions using the instruction-synthesizer model
    outputs = llm(prompt, max_tokens=400, echo=False, stop=['</s>'])
    
    # Extract the generated text from the response
    pred = outputs['choices'][0]['text']
    return parse_pred(pred)

# Get pairs and format them into yaml with indentation
def obtain_pairs(context):
    instruction_response_pairs = get_instruction_response_pairs(context)
    result = """version: 2\ntask_description: [insert based on your context]\ncreated_by: [insert name]\nseed_examples:\n"""
    for index, pair in enumerate(instruction_response_pairs):
        question = LiteralScalarString(f'"{pair["Q"]}"')
        answer = LiteralScalarString(f'"{pair["A"]}"')
        result += f"  - question: {question}\n    answer: {answer}\n"
    file_path = "taxonomy.yaml"
    result += f"document:\n  repo: https://github.com/[insert rest of your fork]\n  commit: [insert commit]\n  patterns:\n    - [insert list of markdown files]"
    with open(file_path, "w") as text_file:
        text_file.write(result)
    return [result, file_path]

# Define the Gradio interface
iface = gr.Interface(
    fn=obtain_pairs,
    inputs=gr.Textbox(interactive=True, label="Context Input"),
    outputs=[
        gr.Textbox(interactive=True, label="Taxonomy Output"),
        gr.File(label="Download Taxonomy Output")
    ],
    title="Instruction Synthesizer for InstructLab",
    description="Input a context and get llm generated instruction-response pairs in a (somewhat) taxonomy acceptable YAML format."
)

# Launch the interface
iface.launch()
