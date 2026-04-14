Propaganda Pipeline
===================

**Propaganda Pipeline** is an open-source Python library for the automatic
detection of propaganda techniques in text. It combines large language models,
consistency testing, and an LLM-as-a-judge consolidation stage to deliver
robust, evidence-backed results across 23 persuasion techniques.

The library is free to use and distributed under the MIT license.

----

Highlights
----------

.. raw:: html

   <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;margin-bottom:1.5rem;">

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>23 Persuasion Techniques</strong><br/>
   Detects the full inventory defined in the JRC annotation guidelines, from
   Loaded Language and Name Calling to False Dilemma and Slippery Slope.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>LLM-as-a-Judge Consolidation</strong><br/>
   A dedicated judge module scores and ranks candidate detections, resolving
   conflicts and filtering out weak signals before producing the final answer.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>Consistency Testing</strong><br/>
   Runs the analysis pipeline multiple times and aggregates results by
   detection ratio, ensuring only reproducible detections are reported.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>Span-level Evidence</strong><br/>
   Every detection comes with verbatim text spans extracted from the source,
   a confidence score, and a plain-language rationale summary.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>HTML Visualizations</strong><br/>
   Built-in rendering highlights detected spans directly in the source text,
   with color-coded badges per technique for easy inspection in notebooks.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>DSPy + OpenAI Backend</strong><br/>
   Built on DSPy for structured LLM programming. Compatible with any
   OpenAI-compatible model endpoint.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>Modular & Extensible</strong><br/>
   Each technique is an independent <code>TechniqueRunner</code> subclass.
   You can run all 23 detectors or select a custom subset for your use case.
   </div>

   <div style="background:#ffffff;border:1px solid #c9e4f7;border-radius:8px;padding:1rem 1.2rem;">
   <strong>Structured Pydantic Outputs</strong><br/>
   All LLM outputs are validated through Pydantic schemas, guaranteeing
   well-formed, type-safe candidate dicts on every run.
   </div>

   </div>

.. raw:: html

   <hr/>

.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   install
   tutorial.ipynb
   api
   techniques
   architecture
   dependencies

