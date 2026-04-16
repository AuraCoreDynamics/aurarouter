# AuraRouter: How Your Tasks are Solved

AuraRouter is not just a "pass-through" for your AI prompts. It uses an intelligent loop to figure out the best way to answer your question. This approach, called **Federated Mixture-of-Experts (FMoE)**, ensures you get high-quality results without the high cost of big cloud models.

---

## 1. The Standard Path: Intent -> Plan -> Execute (IPE)

Most AI tasks follow a simple "Input -> Output" path. AuraRouter adds two smart steps in the middle to save you time and money.

### Step 1: The Classifier (Triage)
Instead of sending your prompt directly to a large model, a tiny, ultra-fast model first "triages" your request. It determines:
- **Intent:** Is this a simple question, a coding task, or a complex analysis?
- **Complexity:** On a scale of 1-10, how hard is this?

### Step 2: The Planner
If your task is complex (e.g., "Build a full website with a database"), AuraRouter doesn't just start typing. A reasoning model creates a **Plan**—a list of logical steps needed to finish the job.

### Step 3: The Worker
Finally, the task is sent to a **Specialist**. If it's a coding task, it goes to a model trained specifically for code. If it's a creative task, it goes to a writing specialist.

**Why do this?**
Traditional methods send every "Hello" to a $20/month cloud giant. AuraRouter sends the "Hello" to a free local model and saves the cloud giant for the tasks that actually need it.

---

## 2. Advanced Reasoning: AuraMonologue

For the hardest tasks (Complexity 8-10), AuraRouter uses **AuraMonologue**. This is like having a "committee of experts" in your computer.

- **The Generator:** Creates the initial answer.
- **The Critic:** Reviews the answer for errors, missing details, or bad logic.
- **The Refiner:** Takes the Critic's feedback and fixes the answer.

The loop continues until the Critic is satisfied. This ensures that even local, smaller models can produce "genius-level" results by checking their own work.

---

## 3. Speed vs. Quality: Speculative Decoding

If you want the speed of a small model but the quality of a large one, AuraRouter uses **Speculative Decoding**.

1. A small "Drafter" model quickly writes a response.
2. A large "Verifier" model checks the draft in the background.
3. If the draft is correct, it's shown to you instantly. If there's a mistake, the verifier fixes it before you even notice.

---

## 4. Local First, Cloud Last

AuraRouter is designed for **Sovereignty**. 

- **Privacy:** It checks your prompts for sensitive data. If it finds any, it forces the task to stay on your local hardware.
- **Cost:** It tracks every cent you save. By using local models for 80% of your tasks, you drastically reduce your monthly AI bill.
- **Control:** You are the "Architect." You decide which models are your favorites and which roles they should play.
