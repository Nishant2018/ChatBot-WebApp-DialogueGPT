from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model_name = 'microsoft/DialoGPT-medium'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
     user_text = request.args.get('msg')
     inputs = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt').to(device)
     outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
     bot_response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
     return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run()
