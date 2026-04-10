# Propaganda Pipeline

Detects 23 propaganda and persuasion techniques in text using LLMs (DSPy + OpenAI).

## Installation

```bash
pip install git+https://github.com/marcelomendoza/propaganda-pipeline.git
```

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

See `propaganda_pipeline_tutorial.ipynb` for a full walkthrough.

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

## Requirements

- Python ≥ 3.10
- OpenAI API key
