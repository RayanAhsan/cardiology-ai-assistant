# Fine-Tuned LLM Cardiology Bot

<img width="1352" height="920" alt="cardiology-bot-project" src="https://github.com/user-attachments/assets/8005ea88-62d9-4f57-8a52-95b1049ed4a9" />



Fine-tuned Meta's **LLaMA-2-7B** model using **PEFT (Parameter-Efficient Fine-Tuning)** and **QLoRA** for the cardiology domain. The model was evaluated with **DeepEval** on 100 prompts, achieving an **average answer relevancy score of 0.83**. A lightweight JavaScript UI was built to allow users to interact with the model seamlessly.  


---


## Features
- Fine-tuned **LLaMA-2-7B** for cardiology-specific Q&A.
- **PEFT + QLoRA** used for efficient training on limited hardware.
- Evaluated using **DeepEval**, scoring 0.83 answer relevancy across 100 evaluation prompts.
- **Custom JavaScript UI** for intuitive model interaction.
- **Caching mechanism** implemented to reduce response latency by ~30 seconds.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/llm-cardiology-bot.git
cd llm-cardiology-bot

pip install -r requirements.txt
