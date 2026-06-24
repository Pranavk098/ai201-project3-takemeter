"""TakeMeter deployed interface: paste an F1 comment, get the predicted register + confidence.

Run:  python app.py     (opens a local web UI)
Requires the fine-tuned model in results/ft_model/ (produced by scripts/finetune_eval.py).
"""
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "results/ft_model"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def classify(text):
    if not text or not text.strip():
        return {}
    inp = tok(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits[0], dim=-1)
    return {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="F1 comment",
                      placeholder="Paste a Formula 1 Reddit comment..."),
    outputs=gr.Label(num_top_classes=4, label="Predicted register (with confidence)"),
    title="TakeMeter — F1 Discourse Classifier",
    description="Fine-tuned DistilBERT. Labels: analysis / hot_take / reaction / joke.",
    examples=[
        ["Leclerc lost that on strategy not pace - he was 0.3s/lap quicker on mediums but Ferrari left him out 4 laps too long and the undercut was gone."],
        ["Verstappen is the most overrated champion in history, put him in a midfield car and he's nothing."],
        ["NOOO not again Ferrari I genuinely can't watch this team anymore"],
        ["Ferrari's strategy department was last seen taking orders from a magic 8-ball."],
    ],
)

if __name__ == "__main__":
    demo.launch()
