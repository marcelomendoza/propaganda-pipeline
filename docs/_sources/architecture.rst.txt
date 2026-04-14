Class Diagram
=============

The diagram below shows the main classes of Propaganda Pipeline and their
relationships. The 23 ``TechniqueRunner`` subclasses are grouped for clarity.

.. raw:: html

    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ startOnLoad: true, theme: 'default' });
    </script>

    <div class="mermaid">
    classDiagram
        direction TB

        class Configuracion {
            +model_name : str
            +api_key : str
            +adapter : Any
            +setup()
        }

        class TechniqueRunner {
            &lt;&lt;abstract&gt;&gt;
            +name : str
            +signature : Type[Signature]
            +run(texto) Dict
            +postprocess(salida_obj) Dict
        }

        class Detectors {
            &lt;&lt;23 subclasses&gt;&gt;
            RepetitionRunner
            ExaggerationRunner
            ObfuscationRunner
            LoadedLanguageRunner
            WhataboutismRunner
            KairosRunner
            ConversationKillerRunner
            SlipperyRunner
            SloganRunner
            AppealToValuesRunner
            RedHerringRunner
            StrawmanRunner
            FearPrejudiceRunner
            AuthorityRunner
            BandwagonRunner
            CastingDoubtRunner
            FlagWavingRunner
            SmearPoisoningRunner
            TuQuoqueRunner
            GuiltByAssociationRunner
            NameCallingRunner
            CausalOversimplificationRunner
            FalseDilemmaRunner
        }

        class Analisis {
            +runners : List[TechniqueRunner]
            +run(texto) List[Dict]
        }

        class ConsistencyModule {
            +trials : int
            +threshold : float
            +run(texto) List[Dict]
        }

        class JudgeModule {
            &lt;&lt;dspy.Module&gt;&gt;
            +forward(question, candidates, rubric) JudgeOutputNormalized
        }

        class SynthesizeModule {
            &lt;&lt;dspy.Module&gt;&gt;
            +forward(question, topk) SynthesisOutputNormalized
        }

        class SelectJudgeCandidatesModule {
            &lt;&lt;dspy.Module&gt;&gt;
            +forward(final_answer, candidates) SelectionOutputNormalized
        }

        TechniqueRunner <|-- Detectors
        Analisis "1" --> "1..*" TechniqueRunner : instantiates
        ConsistencyModule --> Analisis : wraps
        JudgeModule ..> SynthesizeModule : feeds topk
        SynthesizeModule ..> SelectJudgeCandidatesModule : feeds final_answer
        Configuracion ..> Analisis : configures DSPy
    </div>
