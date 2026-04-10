# Propaganda Pipeline

Detects 23 propaganda and persuasion techniques in text using LLMs (DSPy + OpenAI).

## Requirements

- Python ≥ 3.10
- OpenAI API key

## Setup on a new machine

**1. Clone the repo**

```bash
git clone https://github.com/marcelomendoza/propaganda-pipeline.git
cd propaganda-pipeline
```

**2. Install the package and its dependencies**

```bash
pip install -e .
```

**3. Set your OpenAI API key**

```bash
export OPENAI_API_KEY="sk-..."
```

Add that line to your `~/.bashrc` or `~/.zshrc` to make it permanent.

**4. Open the documentation**

```bash
# Linux
xdg-open docs/index.html
# Mac
open docs/index.html
```

The docs include the full API reference and a usage tutorial.

## Quick start

```python
import os
from propaganda_pipeline import Configuracion, Analisis, RunJuez, run_consistency

cfg = Configuracion(model_name="gpt-4o-2024-08-06", api_key=os.environ["OPENAI_API_KEY"])
cfg.setup()

text = "Either you support our policy completely or you want the country to fail."

analisis = Analisis()
candidates = run_consistency(analisis, text, trials=5, threshold=1.0)
result = RunJuez(text, candidates)

print(result["synthesis"]["final_answer"])
print(result["selected_models_posthoc"]["selected_models"])
```

See `propaganda_pipeline_tutorial.ipynb` or `docs/tutorial.html` for a full walkthrough.

## Detected techniques (23)

| Technique | Description |
|-----------|-------------|
| REPETITION | Persuasion through repetition |
| EXAGERATION | Hyperbole or minimization |
| OBFUSCATION | Intentional vagueness |
| LOADED_LANGUAGE | Emotionally charged terms |
| WHATABOUTISM | Deflecting by pointing to others |
| KAIROS | Exploiting timing/urgency |
| KILLER | Conversation killers |
| SLIPPERY | Slippery slope fallacy |
| SLOGAN | Brief striking phrases as sole argument |
| VALUES | Appeal to abstract values |
| RED_HERRING | Introducing irrelevant information |
| STRAWMAN | Misrepresenting someone's position |
| FEAR | Appeal to fear or prejudice |
| AUTHORITY | Fallacious appeal to authority |
| BANDWAGON | Appeal to popularity |
| CASTING_DOUBT | Attacking credibility instead of argument |
| FLAG_WAVING | Appeal to group pride/identity |
| SMEAR_POISONING | Poisoning the well / reputation attacks |
| TU_QUOQUE | "You do it too" / appeal to hypocrisy |
| GUILT_BY_ASSOCIATION | Discrediting by association |
| NAME_CALLING | Derogatory labeling |
| CAUSAL_OVERSIMPLIFICATION | Attributing complex outcomes to a single cause |
| FALSE_DILEMMA | Presenting only two options when more exist |
