# propaganda_pipeline.py

import json
from dataclasses import dataclass
## new
from copy import deepcopy
from IPython.display import display as _display, HTML as _HTML
##
from typing import Any, Dict, List, Optional, Type, Literal, Tuple
import dspy
import pydantic
from pydantic import BaseModel, Field, constr, conint, confloat, field_validator, ValidationError
import time
import re
import unicodedata
import html as _html
from statistics import mean

# =========================================================
# 1) Clase Configuracion
# =========================================================

@dataclass
class Configuracion:
    """Sets up DSPy with a language model and a JSONAdapter.

    Example:
        cfg = Configuracion(
            model_name="openai/gpt-4o-2024-08-06",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        cfg.setup()

    Attributes:
        model_name: OpenAI model identifier (e.g. ``"gpt-4o-2024-08-06"``).
        api_key: OpenAI API key used to authenticate requests.
        adapter: Optional DSPy adapter; defaults to ``dspy.JSONAdapter()``.
    """
    model_name: str
    api_key: str
    adapter: Optional[Any] = None

    def setup(self):
        """Initialises the LM and adapter in DSPy and returns the configured LM.

        Returns:
            dspy.LM: The language model configured in the active DSPy settings.
        """
        lm = dspy.LM(f"openai/{self.model_name}", api_key=self.api_key)

        # Adapter por defecto: JSONAdapter
        adapter = self.adapter or dspy.JSONAdapter()

        # Compatibilidad con distintas versiones de DSPy
        if hasattr(dspy, "settings") and hasattr(dspy.settings, "configure"):
            dspy.settings.configure(lm=lm, adapter=adapter)
        else:
            dspy.configure(lm=lm, adapter=adapter)

        return lm


# =========================================================
# 2) Infraestructura genérica de técnicas
# =========================================================

class TechniqueRunner:
    """Abstract base class for all propaganda-technique detectors.

    Each subclass must declare the class-level attributes ``name`` and
    ``signature`` and implement :meth:`postprocess`.

    Attributes:
        name: Unique technique identifier used as the ``model`` key in
            candidate dicts (e.g. ``"REPETITION"``).
        signature: DSPy Signature class that encodes the detection prompt.
    """

    name: str
    signature: Type[dspy.Signature]

    def __init__(self):
        """Instantiates the DSPy predictor from the subclass ``signature``."""
        self.predictor = dspy.Predict(self.signature)

    def run(self, texto: str) -> Dict[str, Any]:
        """Runs the detector on *texto* and returns a normalised candidate dict.

        Args:
            texto: Source text to analyse.

        Returns:
            A dict containing at least the keys ``model``, ``answer``,
            ``confidence``, ``span``, ``rationale_summary``, and ``raw``.
        """
        pred = self.predictor(texto=texto)
        return self.postprocess(pred.salida)

    def postprocess(self, salida_obj: BaseModel) -> Dict[str, Any]:
        """Converts the Pydantic output from the LLM into a standard candidate dict.

        Args:
            salida_obj: Validated Pydantic object returned by the LLM predictor.

        Returns:
            A dict with at minimum the keys: ``model``, ``answer``,
            ``confidence``, ``span``, ``rationale_summary``, and ``raw``.

        Raises:
            NotImplementedError: Must be implemented by every subclass.
        """
        raise NotImplementedError


# =========================================================
# 3) Técnicas
# =========================================================

# =========================================================
# Técnica REPETITION
# =========================================================


class Conteo(pydantic.BaseModel):
    """(fragment, count) pair quantifying a repeated unit in the text.

    Attributes:
        fragmento: The repeated text fragment (max 120 chars).
        veces: Approximate number of occurrences in the source text.
    """

    fragmento: pydantic.constr(max_length=120)
    veces: int


class JuicioRepetition(pydantic.BaseModel):
    """LLM output for the REPETITION detector.

    Key fields: ``usa_repetition`` (Sí/No binary verdict),
    ``fragmentos_repetidos`` (repeated n-grams), ``conteos_aprox``
    (per-fragment occurrence counts), ``tipo_repeticion`` (rhetorical
    category), ``citas`` (up to 3 verbatim quotes), and ``confidence``.
    """
    usa_repetition: Literal["Sí", "No"]
    claim: Optional[pydantic.constr(max_length=300)] = None
    fragmentos_repetidos: List[pydantic.constr(max_length=120)]
    conteos_aprox: List[Conteo]
    tipo_repeticion: List[
        Literal["anáfora", "epífora", "eslogan", "hashtag", "estribillo", "historia", "imagen"]
    ]
    justificacion: pydantic.constr(max_length=300)
    citas: List[pydantic.constr(max_length=120)]
    dudas_o_limitaciones: pydantic.constr(max_length=300)
    confidence: pydantic.confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)


class DetectaRepetition(dspy.Signature):
    """
    ROL: Eres un analista de argumentos. Determina si el TEXTO usa “Repetition”: un argumento centrado en repetir la misma palabra, frase, consigna, historia o imagen con la esperanza de persuadir por mera repetición.

    ENTRADA
    TEXTO: <texto entre comillas>

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español.
    - Si falta info, usa null o [].
    - Cada string ≤ 300 caracteres. Las citas ≤ 120.
    - No inventes ni reformules citas; extráelas literalmente del TEXTO (máx. 3).
    - Para detectar repetición, considera iguales mayúsculas/minúsculas y minimiza la puntuación; ignora stopwords sueltas y nombres propios necesarios para el tema.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas el span. Manténlo en el idioma original.

    CRITERIOS ESTRICTOS (para “Sí”: deben cumplirse TODOS)
    1) Hay enunciado/posición, llamado o respuesta identificable.
    2) La fuerza persuasiva descansa principalmente (≥50%) en repetir una unidad textual (palabra/frase/eslogan/hashtag/estribillo/metáfora/relato/imagen).
    3) Existe repetición significativa:
       (a) ≥3 ocurrencias no triviales distribuidas en el texto, o
       (b) anáfora/epífora/estribillo/eslogan repetido ≥2 veces en posiciones estratégicas (inicio/cierre/entre párrafos).
    4) Las razones sustantivas (datos, mecanismos, comparaciones precisas) están ausentes o en segundo plano frente a la repetición.

    NO CONFUNDIR
    - Repetición funcional (términos técnicos, nombres propios indispensables, marcadores de lista, estribillos puramente formales).
    - Parafraseo sin repetición léxica clara.
    - Énfasis puntual (1–2 repeticiones breves) sin rol persuasivo central.

    INSTRUCCIONES DE ANÁLISIS
    - Detecta n-gramas (1–5 palabras) repetidos exactos o casi exactos; señala los 1–3 más salientes con conteo aproximado.
    - Trata eslóganes/hashtags/refranes como unidades.
    - Considera “repetición de imagen/relato” cuando la misma escena/metáfora reaparece varias veces.
    - Explica en breve por qué la repetición sí/no es el soporte principal (en la clave "justificacion", ≤300).
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Repetition” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de repetición es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "Hurtlocker deserves an Oscar. Other films have potential, but they do not deserve an Oscar like Hurtlocker does. The other movies may deserve an honorable mention but Hurtlocker deserves the Oscar."

    Traza de decisión resumida (no imprimir en la salida):
    - Normalización y filtrado ligero: minúsculas, puntuación mínima, stopwords sueltas ignoradas.
    - Detección de n-gramas: "hurtlocker deserves an oscar" aparece ≈3 veces (variantes cercanas), es el n-grama más saliente.
    - Criterios:
      C1 (enunciado/posición): ✓ (“Hurtlocker merece el Oscar”).
      C2 (soporte principal por repetición): ✓ la persuasión descansa en repetir el estribillo.
      C3 (significancia): ✓ (≥3 ocurrencias del estribillo).
      C4 (razones sustantivas ausentes/secundarias): ✓ (no hay datos ni comparación precisa).
    - Guardrails: citas literales ≤120; strings ≤300.
    - Confidence: la evidencia es muy clara (múltiples repeticiones clave, sin argumentos alternativos) y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto, por ejemplo 0.95 (alta claridad de evidencia + alta seguridad interna).

    Salida esperada (solo JSON):
    {
      "usa_repetition": "Sí",
      "claim": "Hurtlocker merece el Oscar",
      "fragmentos_repetidos": ["Hurtlocker", "Oscar", "deserves"],
      "conteos_aprox": [
        {"fragmento": "Hurtlocker", "veces": 3},
        {"fragmento": "Oscar", "veces": 3},
        {"fragmento": "deserves", "veces": 2}
      ],
      "tipo_repeticion": ["estribillo"],
      "justificacion": "El texto repite la frase 'Hurtlocker deserves an Oscar' en varias ocasiones, lo que constituye el soporte principal del argumento, sin ofrecer razones sustantivas adicionales.",
      "citas": [
        "Hurtlocker deserves an Oscar.",
        "but they do not deserve an Oscar like Hurtlocker does.",
        "Hurtlocker deserves the Oscar."
      ],
      "dudas_o_limitaciones": "El texto no proporciona un enunciado o posición más allá de la repetición de la frase.",
      "confidence": 0.95,
      "justificacion_confidence": "La evidencia es muy clara (múltiples repeticiones clave, sin argumentos alternativos) y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, comillas incluidas")
    salida: JuicioRepetition = dspy.OutputField(
        desc="Devuelve ÚNICAMENTE este objeto JSON válido en español. No agregues texto fuera del JSON."
    )


class RepetitionRunner(TechniqueRunner):
    """Detects the REPETITION propaganda technique.

    Identifies whether a text persuades primarily through repetition of words,
    phrases, slogans, or images rather than substantive reasoning.

    ``postprocess`` returns a candidate dict with:
        model: ``"REPETITION"``
        answer: ``"Sí"`` or ``"No"``
        confidence: float in [0, 1]
        span: up to 3 verbatim quotes or repeated fragments
        rationale_summary: brief justification from the LLM
        raw: full ``JuicioRepetition`` model dump
    """
    name = "REPETITION"
    signature = DetectaRepetition

    def postprocess(self, salida_obj: JuicioRepetition) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        # Normaliza answer a Sí/No en el formato que usa tu juez
        answer = salida_obj.usa_repetition

        # Span: preferimos citas o fragmentos repetidos (ajustable)
        span = []
        if salida_obj.citas:
            span = salida_obj.citas[:3]
        elif salida_obj.fragmentos_repetidos:
            span = salida_obj.fragmentos_repetidos[:3]

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": salida_obj.justificacion,
            "confidence": float(salida_obj.confidence),
            "span": span,
            # Puedes agregar campos extra si quieres:
            "raw": salida_obj.model_dump()
        }
    
# =========================================================
# Técnica EXAGERATION / MINIMISATION
# =========================================================

class JuicioEM(pydantic.BaseModel):
    """LLM output for the EXAGGERATION / MINIMISATION detector.

    Key fields: ``is_exaggeration_or_minimisation`` (Sí/No verdict),
    ``subtipo`` (Exageración / Minimización / Ambas), ``puntaje_exageracion``
    and ``puntaje_minimizacion`` (independent [0,1] scores),
    ``proporcionalidad`` (language-to-fact ratio assessment),
    ``citas`` (up to 3 verbatim quotes), and ``confidence``.
    """
    is_exaggeration_or_minimisation: Literal["Sí", "No"]
    subtipo: Literal["Exageración", "Minimización", "Ambas", "No aplica"]
    justificacion_breve: pydantic.constr(max_length=300)
    citas: List[pydantic.constr(max_length=120)]
    marcadores_exageracion: List[pydantic.constr(max_length=120)]
    marcadores_minimizacion: List[pydantic.constr(max_length=120)]
    objeciones_ignoradas: Optional[pydantic.constr(max_length=300)] = None
    proporcionalidad: Literal["Desproporcionado", "Aproximadamente proporcional", "Incierto"]
    puntaje_exageracion: pydantic.confloat(ge=0.0, le=1.0)
    puntaje_minimizacion: pydantic.confloat(ge=0.0, le=1.0)
    reformulacion_neutra: pydantic.constr(max_length=300)
    confidence: pydantic.confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)


class DetectaExaggerationMinimisation(dspy.Signature):
    """
    ROL: Eres un analista de argumentos. Detecta si el TEXTO usa “Exaggeration or Minimisation”:
    exagerar (hipérboles/superlativos/absolutos) o minimizar (atenuadores/eufemismos/“no es para tanto”)
    para persuadir, desplazando razones sustantivas.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español.
    - Si falta info, usa null o [].
    - Cada string ≤ 300 caracteres. Citas ≤ 120.
    - No inventes ni reformules citas; extráelas literalmente del TEXTO (máx. 3).
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS ESTRICTOS (para “Sí”)
    1) Hay enunciado/posición o respuesta a una pregunta.
    2) La fuerza persuasiva descansa principalmente (≥50%) en exagerar o minimizar
       (superlativos/absolutos/hipérboles; o “solo”, “apenas”, “no es para tanto”…).
    3) Ausencia o marginalidad de razones sustantivas (datos, mecanismos, comparaciones, contexto proporcional)
       o se ignoran objeciones relevantes.

    SEÑALES (no exhaustivas)
    - Exageración: “el mejor/peor”, “siempre/nunca”, “todos/nadie”, “garantizado”, cuantificaciones infladas sin soporte.
    - Minimización: “solo un errorcito”, “apenas pasó”, “una broma”, “no importa”, “nada grave”, eufemismos.
    - Desproporción entre lenguaje y hechos; omite contexto relevante.

    DISTINCIONES (NO contar como Sí)
    - Énfasis retórico con evidencia suficiente; metáfora ilustrativa con soporte; estimaciones con fuentes claras; ironía explícita.

    INSTRUCCIONES DE ANÁLISIS
    - Trabaja con el texto tal cual; normaliza mínimamente (insensible a mayúsculas/minúsculas; puntuación ligera).
    - Extrae 1–3 citas literales clave (≤120 c/u) que evidencien exageración/minimización.
    - Señala marcadores de exageración/minimización (palabras/expresiones gatillo).
    - Evalúa proporcionalidad del lenguaje respecto del contenido disponible.
    - Si hay objeciones plausibles omitidas, menciónalas en "objeciones_ignoradas" (≤300).
    - Asigna puntajes [0,1] a exageración y minimización (no tienen que sumar 1).

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Exaggeration” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de Exaggeration es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.


    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "From the seminaries, to the clergy, to the bishops, to the cardinals, homosexuals are present at all levels, by the thousand"

    Traza de decisión resumida (no imprimir en la salida):
    - Identificación de posición (C1): se afirma presencia amplia en todos los niveles.
    - Señal/soporte principal (C2): cuantificación inflada “by the thousand” funciona como hipérbole central.
    - Sustento (C3): no hay datos comparativos ni fuentes; se ignoran posibles matices (definiciones, cifras, períodos).
    - Marcadores: “by the thousand”; construcción de totalidad “at all levels”.
    - Proporcionalidad: desproporcionado respecto a la evidencia presentada (ninguna).
    - Guardrails: citas literales ≤120; strings ≤300; salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "is_exaggeration_or_minimisation": "Sí",
      "subtipo": "Exageración",
      "justificacion_breve": "El texto afirma presencia en todos los niveles 'por miles', cuantificación inflada sin soporte que funciona como hipérbole persuasiva.",
      "citas": [
        "homosexuals are present at all levels, by the thousand"
      ],
      "marcadores_exageracion": [
        "by the thousand",
        "at all levels"
      ],
      "marcadores_minimizacion": [],
      "objeciones_ignoradas": null,
      "proporcionalidad": "Desproporcionado",
      "puntaje_exageracion": 0.8,
      "puntaje_minimizacion": 0.0,
      "reformulacion_neutra": "Se afirma presencia de homosexuales en varios niveles, sin cuantificar con precisión ni fuentes.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    ─────────────────────────────────────────────────────────────────────────────
    SALIDA (SOLO JSON) exactamente con estas claves.
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece (en comillas si aplica)")
    salida: JuicioEM = dspy.OutputField(
        desc="Objeto JSON válido que cumple el esquema y las restricciones de longitud."
    )


class ExaggerationRunner(TechniqueRunner):
    """Detects the EXAGGERATION / MINIMISATION propaganda technique.

    Identifies whether a text persuades primarily through hyperboles,
    superlatives, and absolute claims (exaggeration) or downplaying
    language (minimisation), instead of substantive reasoning.

    ``postprocess`` returns a candidate dict with:
        model: ``"EXAGERATION"``
        answer: ``"Sí"`` or ``"No"``
        confidence: float in [0, 1]
        span: up to 3 verbatim quotes or exaggeration/minimisation markers
        rationale_summary: brief justification including subtype and proportionality
        raw: full ``JuicioEM`` model dump
    """
    name = "EXAGERATION"
    signature = DetectaExaggerationMinimisation

    def postprocess(self, salida_obj: JuicioEM) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        # answer en el formato del juez
        answer = salida_obj.is_exaggeration_or_minimisation

        # span: preferimos citas literales; si no hay, usamos marcadores
        span = []
        if salida_obj.citas:
            span = salida_obj.citas[:3]
        else:
            markers = (salida_obj.marcadores_exageracion or []) + (salida_obj.marcadores_minimizacion or [])
            span = markers[:3]

        # rationale_summary compacto pero informativo
        rationale_parts = [salida_obj.justificacion_breve]

        # Añadimos subtipo y proporcionalidad solo si aporta
        if salida_obj.subtipo and salida_obj.subtipo != "No aplica":
            rationale_parts.append(f"Subtipo: {salida_obj.subtipo}.")
        if salida_obj.proporcionalidad:
            rationale_parts.append(f"Proporcionalidad: {salida_obj.proporcionalidad}.")

        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,
            # Campos extra útiles para debug / trazabilidad:
            "labels": {
                "subtipo": salida_obj.subtipo,
                "puntaje_exageracion": float(salida_obj.puntaje_exageracion),
                "puntaje_minimizacion": float(salida_obj.puntaje_minimizacion),
                "proporcionalidad": salida_obj.proporcionalidad,
                "objeciones_ignoradas": salida_obj.objeciones_ignoradas,
            },
            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica OBFUSCATION / INTENTIONAL VAGUENESS / CONFUSION
# =========================================================

class AmbiguousTerm(pydantic.BaseModel):
    """A vague or undefined key term together with its plausible interpretations.

    Attributes:
        term: The ambiguous term extracted from the text (max 120 chars).
        possible_meanings: Up to 2 plausible meanings suggested by the LLM.
    """
    term: pydantic.constr(max_length=120)
    possible_meanings: List[pydantic.constr(max_length=120)] = Field(default_factory=list)

class NonAnswerEvasion(pydantic.BaseModel):
    """Flags whether the text evades answering the question under discussion.

    Attributes:
        value: ``True`` if the text uses vagueness to avoid a direct answer.
        evidence: Optional verbatim excerpt supporting the evasion flag.
    """
    value: bool
    evidence: Optional[pydantic.constr(max_length=120)] = None

class ObfuscationJudgment(pydantic.BaseModel):
    """LLM output for the OBFUSCATION / INTENTIONAL VAGUENESS detector.

    Key fields: ``uses_obfuscation`` (Sí/No/Indeterminado verdict),
    ``vague_markers`` (weasel-word signals), ``ambiguous_terms``
    (undefined key terms with possible meanings), ``non_answer_evasion``
    (whether the text evades direct answers), ``quotes`` (verbatim
    excerpts), and ``confidence``.
    """
    uses_obfuscation: Literal["Sí", "No", "Indeterminado"]
    justification: pydantic.constr(max_length=300)
    vague_markers: List[pydantic.constr(max_length=120)]
    ambiguous_terms: List[AmbiguousTerm]
    non_answer_evasion: NonAnswerEvasion
    undefined_keys: List[pydantic.constr(max_length=120)]
    quotes: List[pydantic.constr(max_length=120)]
    confidence: pydantic.confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)


class DetectaObfuscacion(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Determina si el TEXTO usa “Obfuscation, Intentional Vagueness, Confusion”.

    GUARDRAIL
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si falta info, usa null o [].
    - Cada string ≤ 300 caracteres. Las citas ≤ 120. No inventes ni reformules citas.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS (TODOS para “Sí”)
    - Hay enunciado/posición o respuesta a una pregunta.
    - La justificación descansa principalmente en vaguedad/ambigüedad (weasel words, términos sin definir,
      eufemismos técnicos, cuantificadores difusos, frases no falsables).
    - La vaguedad impide evaluar/verificar la conclusión o elude responder (non-answer).
    - Faltan definiciones operativas/métricas/criterios o ejemplos concretos para términos clave.

    DISTINCIONES
    - Brevedad con criterios claros, metáfora ilustrativa con soporte, o falta de detalle con criterios verificables ≠ obfuscación.

    SEÑALES (indicativas)
    - “algunos”, “se está trabajando”, “medidas necesarias”, “mejoras significativas”, “muchos”, “varios”, “seguridad”, etc.

    INSTRUCCIONES DE ANÁLISIS
    - Identifica marcadores vagos/ambiguos y términos clave sin definir.
    - Para cada término ambiguo, sugiere 1–2 significados plausibles si el texto no define.
    - Señala si hay evasión de respuesta (non-answer) y aporta evidencia breve (≤120).
    - Extrae hasta 3 citas literales (≤120) si aportan evidencia; si no, usa [].
    - Explica en ≤300 por qué la vaguedad/ambigüedad es (o no) soporte principal.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Obfuscation” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Obfuscation” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "We will hex-develop the blockchain with AI-based interconnectors to maximize ROI."

    Traza de decisión resumida (no imprimir en la salida):
    - Normalización rápida y detección de términos no definidos: “hex-develop”, “AI-based interconnectors”, “maximize ROI”.
    - Comprobación de criterios:
      • Enunciado/posición: ✓ (promesa de acción/futuro).
      • Soporte principal por vaguedad/ambigüedad: ✓ (términos sin definir, eufemismo técnico).
      • Efecto de la vaguedad: ✓ (impide evaluar la afirmación y elude precisiones => posible non-answer).
      • Falta de definiciones/criterios: ✓ (no hay métricas, ejemplos ni especificaciones).
    - Guardrails: sin inventar citas; strings ≤300; si no hay citas útiles, quotes = [].
    - Clasificación prevista: “Sí”, con lista de términos ambiguos y significados plausibles.

    Salida esperada (solo JSON):
    {
      "uses_obfuscation": "Sí",
      "justification": "El texto utiliza términos vagos y técnicos como 'hex-develop', 'AI-based interconnectors' y 'maximizar ROI' sin definirlos claramente, lo que dificulta evaluar la afirmación.",
      "vague_markers": [
        "hex-develop",
        "AI-based interconnectors",
        "maximizar ROI"
      ],
      "ambiguous_terms": [
        {
          "term": "hex-develop",
          "possible_meanings": [
            "Desarrollo en hexadecimal",
            "Desarrollo avanzado"
          ]
        },
        {
          "term": "AI-based interconnectors",
          "possible_meanings": [
            "Conectores basados en inteligencia artificial",
            "Interconexiones automatizadas"
          ]
        },
        {
          "term": "maximizar ROI",
          "possible_meanings": [
            "Aumentar el retorno de inversión",
            "Optimizar beneficios"
          ]
        }
      ],
      "non_answer_evasion": {
        "value": true,
        "evidence": "El uso de términos vagos y técnicos sin definición impide una evaluación clara."
      },
      "undefined_keys": [
        "hex-develop",
        "AI-based interconnectors",
        "maximizar ROI"
      ],
      "quotes": [],
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    ─────────────────────────────────────────────────────────────────────────────

    SALIDA (SOLO JSON) exactamente con estas claves.
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: ObfuscationJudgment = dspy.OutputField(
        desc="Devuelve ÚNICAMENTE el objeto JSON válido conforme al esquema."
    )

class ObfuscationRunner(TechniqueRunner):
    """Detects the OBFUSCATION / INTENTIONAL VAGUENESS / CONFUSION technique.

    Identifies whether persuasion relies primarily on vague, undefined, or
    evasive language that prevents the audience from evaluating the claim.

    ``postprocess`` returns a candidate dict with:
        model: ``"OBFUSCATION"``
        answer: ``"Sí"``, ``"No"``, or ``"Indeterminado"``
        confidence: float in [0, 1]
        span: up to 3 verbatim quotes, vague markers, or undefined key terms
        rationale_summary: brief justification from the LLM
        raw: full ``ObfuscationJudgment`` model dump
    """
    name = "OBFUSCATION"
    signature = DetectaObfuscacion

    def postprocess(self, salida_obj: ObfuscationJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        # answer: mantenemos "Indeterminado" para que tu rúbrica lo trate como ambiguo
        answer = salida_obj.uses_obfuscation

        # span: prioriza quotes literales, si no hay usa marcadores/keys/terms
        span = []
        if salida_obj.quotes:
            span = salida_obj.quotes[:3]
        else:
            # mezcla señales cortas de evidencia
            markers = list(salida_obj.vague_markers or [])
            keys = list(salida_obj.undefined_keys or [])
            terms = [t.term for t in (salida_obj.ambiguous_terms or [])]

            merged = []
            for x in markers + keys + terms:
                if x and x not in merged:
                    merged.append(x)
            span = merged[:3]

        # rationale_summary: compacto y central
        rationale_summary = salida_obj.justification.strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            # labels opcionales (útiles para debug o para tu consolidación)
            "labels": {
                "non_answer_evasion": bool(salida_obj.non_answer_evasion.value),
                "non_answer_evidence": salida_obj.non_answer_evasion.evidence,
                "num_vague_markers": len(salida_obj.vague_markers or []),
                "num_ambiguous_terms": len(salida_obj.ambiguous_terms or []),
                "num_undefined_keys": len(salida_obj.undefined_keys or []),
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica LOADED_LANGUAGE
# =========================================================

class LoadedLanguageJudgment(BaseModel):
    """LLM output for the LOADED LANGUAGE detector.

    Key fields: ``is_loaded_language`` (Sí/No verdict), ``charged_terms``
    (emotionally loaded words), ``quotes`` (up to 3 verbatim excerpts),
    ``emotions`` (emotions invoked), ``valence`` (positive/negative/mixed),
    ``reasoning_presence`` and ``evidence_presence`` (quality of substantive
    support), ``cooccurring_techniques``, and ``confidence``.
    """
    is_loaded_language: Literal["Sí", "No"]
    claim: Optional[constr(max_length=300)] = None
    charged_terms: List[constr(max_length=120)] = Field(default_factory=list)
    quotes: List[constr(max_length=120)] = Field(default_factory=list)
    emotions: List[constr(max_length=120)] = Field(default_factory=list)
    valence: Literal["positiva", "negativa", "mixta", "neutra"]
    reasoning_presence: Literal["ninguna", "escasa", "moderada", "sólida"]
    evidence_presence: Literal["ninguna", "escasa", "moderada", "sólida"]
    cooccurring_techniques: List[constr(max_length=120)] = Field(default_factory=list)
    explanation: constr(max_length=300)
    decision_rule: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

    # Guardrail: limitar a máx. 3 citas textuales
    @field_validator("quotes")
    @classmethod
    def _limit_quotes(cls, v: List[str]) -> List[str]:
        return v[:3]


class DetectaLoadedLanguage(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta si el TEXTO usa “Loaded Language (Lenguaje cargado)”.

    GUARDRAIL (SOLO JSON)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si falta info, usa null o [].
    - Cada string ≤ 300 caracteres. Citas ≤ 120. Máx. 3 citas literales. No inventes ni reformules citas.
    - NO incluyas razonamientos paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS (para “Sí”, deben cumplirse TODOS)
    1) Hay enunciado/posición/llamado identificable.
    2) La justificación descansa principalmente (≥50%) en términos cargados (eufemismos/disfemismos,
       adjetivos valorativos, hipérboles, metáforas incendiarias, absolutos).
    3) Las razones sustantivas están ausentes/débiles/subordinadas al efecto emocional.

    DISTINCIONES
    - Name calling cuenta solo si domina la persuasión.
    - Si la apelación central es miedo/orgullo grupal/popularidad, indícalo en “cooccurring_techniques”.
    - Lenguaje vívido con evidencia sólida y dominante → No es Loaded Language.

    SEÑALES
    - Adjetivos intensificadores (“corrupto”, “asqueroso”, “heroico”).
    - Sustantivos moralizantes/estigmatizantes (“parásitos”, “salvadores”).
    - Metáforas bélicas/contagio (“guerra”, “plaga”, “virus”).
    - Eufemismos (“daño colateral”) y disfemismos (“masacre”).
    - Absolutos (“siempre”, “todos”, “nadie”, “jamás”).

     CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “LoadedLanguage” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “LoadedLanguage” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.


    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "How stupid and petty things have become in Washington."

    Traza de decisión resumida (no imprimir en la salida):
    - Normalización ligera (minúsculas/puntuación) sin alterar citas literales.
    - Identificación de términos potencialmente cargados: “stupid”, “petty”.
    - Criterio 1 (enunciado/posición): ✓ hay valoración global de la situación en Washington.
    - Criterio 2 (peso persuasivo en lenguaje cargado): ✓ la evaluación depende de adjetivos valorativos.
    - Criterio 3 (razones/evidencia): ✓ ausencia de evidencia o razonamiento; juicio eminentemente emocional.
    - Diferenciar de técnicas cercanas: no hay apelación principal a orgullo/miedo/popularidad; registrar co-ocurrencias si emergen.
    - Guardrails: ≤3 citas; strings ≤300; no inventar citas; salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "is_loaded_language": "Sí",
      "claim": "Las cosas se han vuelto estúpidas y mezquinas en Washington.",
      "charged_terms": ["estúpidas", "mezquinas"],
      "quotes": ["How stupid and petty things have become in Washington"],
      "emotions": ["frustración", "desdén"],
      "valence": "negativa",
      "reasoning_presence": "ninguna",
      "evidence_presence": "ninguna",
      "cooccurring_techniques": [],
      "explanation": "El texto utiliza términos cargados como 'estúpidas' y 'mezquinas' para describir la situación en Washington, sin proporcionar evidencia o razonamiento sustantivo.",
      "decision_rule": "El uso de términos cargados sin evidencia o razonamiento sustantivo indica lenguaje cargado.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    ─────────────────────────────────────────────────────────────────────────────

    SALIDA (JSON) exactamente con las claves pedidas.
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: LoadedLanguageJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class LoadedLanguageRunner(TechniqueRunner):
    """Detects the LOADED LANGUAGE propaganda technique.

    Identifies whether persuasion relies primarily on emotionally charged
    euphemisms, dysphemisms, intensifying adjectives, or inflammatory
    metaphors rather than substantive evidence.

    ``postprocess`` returns a candidate dict with:
        model: ``"LOADED_LANGUAGE"``
        answer: ``"Sí"`` or ``"No"``
        confidence: float in [0, 1]
        span: up to 3 verbatim quotes or charged terms
        rationale_summary: explanation including emotional valence
        raw: full ``LoadedLanguageJudgment`` model dump
    """
    name = "LOADED_LANGUAGE"
    signature = DetectaLoadedLanguage

    def postprocess(self, salida_obj: LoadedLanguageJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_loaded_language

        # span: prioriza quotes literales, si no hay usa charged_terms
        span = []
        if salida_obj.quotes:
            span = salida_obj.quotes[:3]
        elif salida_obj.charged_terms:
            span = salida_obj.charged_terms[:3]

        # rationale_summary compacto: explanation + valence si aplica
        rationale_parts = [salida_obj.explanation.strip()]
        if salida_obj.valence:
            rationale_parts.append(f"Valencia: {salida_obj.valence}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            # labels opcionales útiles para depuración / consolidación
            "labels": {
                "claim": salida_obj.claim,
                "charged_terms": salida_obj.charged_terms,
                "emotions": salida_obj.emotions,
                "valence": salida_obj.valence,
                "reasoning_presence": salida_obj.reasoning_presence,
                "evidence_presence": salida_obj.evidence_presence,
                "cooccurring_techniques": salida_obj.cooccurring_techniques,
                "decision_rule": salida_obj.decision_rule,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica WHATABOUTISM (SWITCHING TOPIC)
# =========================================================

class FocusShift(BaseModel):
    """Whether the text shifts focus away from the original topic.

    Attributes:
        value: ``True`` if focus is demonstrably shifted to a different topic.
        evidence: Optional verbatim excerpt supporting the shift.
    """
    value: bool
    evidence: Optional[constr(max_length=120)] = None  # "cita breve o null"


class WhataboutismJudgment(BaseModel):
    """LLM output for the WHATABOUTISM / SWITCHING TOPIC detector.

    Key fields: ``is_whataboutism`` (Sí/No verdict), ``original_issue``
    (topic being deflected), ``switching_issue`` (topic introduced instead),
    ``explicit_markers_found`` (verbatim deflection markers), ``focus_shift``
    (whether focus was demonstrably moved), ``accusation_type`` (hypocrisy /
    double standard / etc.), and ``confidence``.
    """
    is_whataboutism: Literal["Sí", "No"]
    verdict: Literal[
        "Sí, hay Switching Topic (Whataboutism)",
        "No, no hay Switching Topic (Whataboutism)"
    ]
    original_issue: Optional[constr(max_length=120)] = None
    switching_issue: Optional[constr(max_length=120)] = None
    explicit_markers_found: List[constr(max_length=120)] = Field(default_factory=list)
    focus_shift: FocusShift
    accusation_type: List[
        Literal["hipocresía/tu_quoque", "doble_estándar", "comparación_desviadora", "otro"]
    ] = Field(default_factory=list)
    reason: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaWhataboutism(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Determina si el TEXTO usa “Switching Topic (Whataboutism)”.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si falta información, usa null o [].
    - Todas las cadenas ≤ 300 caracteres. Citas del TEXTO ≤ 120.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIO ESTRICTO (TODOS para “Sí”)
    1) Tema original identificable que merece respuesta.
    2) Introduce tema acusatorio (hipocresía/doble estándar/“¿y qué hay de…?”).
    3) Desplaza el foco y NO responde sustantivamente el tema original.

    DISTINCIONES
    - Red herring sin acusación de hipocresía ≠ whataboutism.
    - Tu quoque con respuesta sustantiva al fondo → puede NO ser whataboutism.
    - Comparación pertinente que sí responde al fondo → NO es whataboutism.

    SEÑALES
    - “¿y qué hay de…?”, “ustedes hicieron lo mismo”, “doble estándar”, “antes X y ahora Y”, etc.

    INSTRUCCIONES DE ANÁLISIS
    - Identifica con citas breves (≤120) el tema original y el tema desplazado.
    - Localiza marcadores explícitos (“¿y qué hay de…?”, comparaciones desviadoras, etc.).
    - Señala si hay DESPLAZAMIENTO DE FOCO (focus_shift.value) y cita evidencia breve.
    - Clasifica la acusación principal: "hipocresía/tu_quoque", "doble_estándar", "comparación_desviadora" u "otro".
    - Resume en ≤300 por qué SÍ/NO hay whataboutism.

     CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “whataboutism” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “whataboutism” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "Bien sûr, les actions militaires sont toujours horribles et sont d’autant plus condamnables lorsqu’elles
    touchent des populations civiles. Toutefois, l’offensive Russe en Ukraine, est-elle plus que l’invasion
    américaine en Irak en 2003? Ces événements ont eux aussi causé des décès de civils sans pourtant soulever
    l’ire de la bien-pensance occidentale"

    Traza de decisión resumida (no imprimir en la salida):
    - Normalización ligera y lectura del flujo argumental.
    - Tema original detectado (cita breve): "offensive Russe en Ukraine".
    - Tema desplazado/alternativo (cita breve): "invasion américaine en Irak en 2003".
    - Marcador explícito de comparación desviadora (cita ≤120): "est-elle plus que l’invasion américaine en Irak en 2003".
    - Evaluación de foco:
        * ¿Se responde sustantivamente al tema original? → No: se compara para denunciar doble estándar.
        * focus_shift.value = True; focus_shift.evidence = misma cita del marcador.
    - Tipo de acusación principal: "doble_estándar".
    - Guardrails verificados: JSON-only, longitudes, citas literales del TEXTO (sin reformular), español.

    Salida esperada (solo JSON):
    {
      "is_whataboutism": "Sí",
      "verdict": "Sí, hay Switching Topic (Whataboutism)",
      "original_issue": "offensive Russe en Ukraine",
      "switching_issue": "invasion américaine en Irak en 2003",
      "explicit_markers_found": [
        "est-elle plus que l’invasion américaine en Irak en 2003"
      ],
      "focus_shift": {
        "value": true,
        "evidence": "est-elle plus que l’invasion américaine en Irak en 2003"
      },
      "accusation_type": ["doble_estándar"],
      "reason": "El texto compara la ofensiva rusa en Ucrania con la invasión estadounidense en Irak, sugiriendo un doble estándar en la condena de acciones militares, desviando el foco del tema original.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    ─────────────────────────────────────────────────────────────────────────────
    SALIDA (JSON) exactamente con las claves pedidas.
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: WhataboutismJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class WhataboutismRunner(TechniqueRunner):
    """Detects the WHATABOUTISM (Switching Topic) propaganda technique.

    Identifies whether a response deflects from the original topic by
    introducing an accusation of hypocrisy or a double standard, thereby
    avoiding substantive engagement with the original issue.

    ``postprocess`` returns a candidate dict with:
        model: ``"WHATABOUTISM"``
        answer: ``"Sí"`` or ``"No"``
        confidence: float in [0, 1]
        span: up to 3 explicit deflection markers or focus-shift evidence
        rationale_summary: reason summary including accusation type
        raw: full ``WhataboutismJudgment`` model dump
    """
    name = "WHATABOUTISM"
    signature = DetectaWhataboutism

    def postprocess(self, salida_obj: WhataboutismJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_whataboutism

        # span: prioriza marcadores explícitos; si no, evidencia de focus shift;
        # si no, issue original/switching
        span = []
        if salida_obj.explicit_markers_found:
            span = (salida_obj.explicit_markers_found or [])[:3]
        elif salida_obj.focus_shift and salida_obj.focus_shift.evidence:
            span = [salida_obj.focus_shift.evidence][:3]
        else:
            merged = []
            for x in [salida_obj.original_issue, salida_obj.switching_issue]:
                if x and x not in merged:
                    merged.append(x)
            span = merged[:3]

        # rationale_summary: reason + tipo de acusación si existe
        rationale_parts = [salida_obj.reason.strip()]
        if salida_obj.accusation_type:
            rationale_parts.append(f"Acusación: {', '.join(salida_obj.accusation_type)}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "verdict": salida_obj.verdict,
                "original_issue": salida_obj.original_issue,
                "switching_issue": salida_obj.switching_issue,
                "focus_shift": bool(salida_obj.focus_shift.value),
                "focus_shift_evidence": salida_obj.focus_shift.evidence,
                "explicit_markers_found": salida_obj.explicit_markers_found,
                "accusation_type": salida_obj.accusation_type,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica KAIROS (APPEAL TO TIME)
# =========================================================

class KairosJudgment(BaseModel):
    """LLM output for the KAIROS / APPEAL TO TIME detector.

    Key fields: ``is_appeal_to_time`` (Sí/No/Mixto verdict),
    ``actions_or_claims`` (proposed actions or evaluative claims),
    ``timing_devices`` (urgency language such as "now is the time"),
    ``quotes`` (up to 3 verbatim excerpts), ``reasoning`` (brief
    justification), ``counter_indicators`` (signals that weaken the
    verdict), and ``confidence``.
    """
    is_appeal_to_time: Literal["Sí", "No", "Mixto"]
    actions_or_claims: List[constr(max_length=300)] = Field(default_factory=list)
    timing_devices: List[constr(max_length=120)] = Field(default_factory=list)
    quotes: List[constr(max_length=120)] = Field(default_factory=list)
    reasoning: constr(max_length=300)
    counter_indicators: List[constr(max_length=300)] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

    @field_validator("quotes")
    @classmethod
    def _limit_quotes(cls, v: List[str]) -> List[str]:
        return v[:3]


class DetectaKairos(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Determina si el TEXTO usa “Appeal to Time (Kairos)”.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español.
    - Si falta información, usa null o [].
    - Cada string ≤ 300 caracteres. Citas ≤ 120 (máx. 3).
    - No imprimas razonamiento paso a paso; piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS (para “Sí”, TODOS)
    1) Hay propuesta/llamado/posición evaluativa.
    2) La justificación descansa principalmente en urgencia/oportunidad temporal (“ahora”, “última oportunidad”, “ventana que se cierra”, etc.).
    3) Razones sustantivas independientes del tiempo están ausentes/débiles/subordinadas.

    SEÑALES
    - Contadores/fechas límite; léxico de urgencia; “ventana que se cierra”.

    DISTINCIONES
    - Plazos reales con razones materiales dominantes → No es Kairos.
    - Si el peso principal es miedo/popularidad → clasificar como co-ocurrencia (si aplicara).

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Kairos” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Kairos” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism. 
    Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate 
    valley of segregation to the sunlit path of racial justice."

    Traza de decisión resumida (no imprimir en la salida):
    - Normalización y lectura focal: identificar léxico temporal y actos propuestos.
    - Criterio 1 (propuesta/llamado): ✓ “make real the promises…”, “rise from… to…”.
    - Criterio 2 (peso en urgencia/oportunidad temporal): ✓ “Now is the time” repetido y centrado en el “ahora”.
    - Criterio 3 (razones sustantivas independientes del tiempo): ✓ ausentes/implícitas; el soporte persuasivo es el timing.
    - Evidencia textual (máx. 3 citas ≤120): extraer literalmente dos “Now is the time…” como unidades salientes.
    - Guardrails: strings ≤300; citas ≤120; salida SOLO JSON, sin comentarios externos.

    Salida esperada (solo JSON):
    {
      "is_appeal_to_time": "Sí",
      "actions_or_claims": [
        "Make real the promises of democracy",
        "Rise from the dark and desolate valley of segregation to the sunlit path of racial justice"
      ],
      "timing_devices": [
        "Now is the time"
      ],
      "quotes": [
        "Now is the time to make real the promises of democracy",
        "Now is the time to rise from the dark and desolate valley of segregation"
      ],
      "reasoning": "El texto enfatiza la urgencia de actuar ahora para lograr la justicia racial, destacando que no es momento para la gradualidad.",
      "counter_indicators": [],
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    ─────────────────────────────────────────────────────────────────────────────

    SALIDA (JSON) exactamente con las claves pedidas.
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: KairosJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class KairosRunner(TechniqueRunner):
    """Detects the KAIROS / APPEAL TO TIME propaganda technique.

    Identifies whether persuasion rests primarily on urgency or
    opportunity framing ("now is the time", "closing window") rather
    than substantive reasoning independent of timing.

    ``postprocess`` returns a candidate dict with:
        model: ``"KAIROS"``
        answer: ``"Sí"``, ``"No"``, or ``"Mixto"``
        confidence: float in [0, 1]
        span: up to 3 verbatim quotes, timing devices, or claim fragments
        rationale_summary: reasoning with note if counter-indicators are present
        raw: full ``KairosJudgment`` model dump
    """
    name = "KAIROS"
    signature = DetectaKairos

    def postprocess(self, salida_obj: KairosJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_appeal_to_time  # Sí / No / Mixto

        # span: prioriza quotes, luego timing_devices, luego actions_or_claims
        span = []
        if salida_obj.quotes:
            span = (salida_obj.quotes or [])[:3]
        elif salida_obj.timing_devices:
            span = (salida_obj.timing_devices or [])[:3]
        else:
            span = (salida_obj.actions_or_claims or [])[:3]

        # rationale_summary: reasoning + (si Mixto o counter) una nota mínima
        rationale_parts = [salida_obj.reasoning.strip()]
        if salida_obj.counter_indicators:
            rationale_parts.append("Contra-indicios presentes.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "actions_or_claims": salida_obj.actions_or_claims,
                "timing_devices": salida_obj.timing_devices,
                "counter_indicators": salida_obj.counter_indicators,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica CONVERSATION_KILLER (Thought-terminating cliché)
# =========================================================

class SuppressesDiscussion(BaseModel):
    """Whether the text actively discourages further debate.

    Attributes:
        value: ``True`` if the text contains a thought-terminating element.
        evidence: Optional verbatim excerpt demonstrating the suppression.
    """
    value: bool
    evidence: Optional[constr(max_length=120)] = None  # idealmente una cita breve literal

class ReasonsPresent(BaseModel):
    """Degree to which substantive reasoning accompanies the cliché.

    Attributes:
        level: Qualitative assessment: ``"ninguna"``, ``"baja"``,
            ``"media"``, or ``"alta"``.
        evidence: Optional excerpt supporting the level assessment.
    """
    level: Literal["ninguna", "baja", "media", "alta"]
    evidence: Optional[constr(max_length=300)] = None

class ConversationKillerJudgment(BaseModel):
    """LLM output for the CONVERSATION KILLER / THOUGHT-TERMINATING CLICHÉ detector.

    Key fields: ``es_conversation_killer`` (Sí/No verdict),
    ``cliches_detected`` (thought-terminating phrases found),
    ``primary_mechanism`` (closing mechanism type), ``suppresses_discussion``
    (whether discussion is discouraged), ``reasons_present`` (level of
    substantive justification), ``quotes`` (up to 3 verbatim excerpts),
    and ``confidence``.
    """
    es_conversation_killer: Literal["Sí", "No"]
    cliches_detected: List[constr(max_length=120)] = Field(default_factory=list)
    quotes: List[constr(max_length=120)] = Field(default_factory=list)
    primary_mechanism: Literal[
        "imperativo_de_cierre", "fatalismo", "antiintelectualismo",
        "identitario", "relativismo", "otro"
    ]
    suppresses_discussion: SuppressesDiscussion
    reasons_present: ReasonsPresent
    topic_identified: Optional[constr(max_length=300)] = None
    explanation: constr(max_length=300)
    suggested_reframe: List[constr(max_length=120)] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

    @field_validator("quotes")
    @classmethod
    def _limit_quotes(cls, v: List[str]) -> List[str]:
        return v[:3]


class DetectaConversationKiller(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta si el TEXTO usa “Conversation Killer (Thought-terminating Cliché)”.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español.
    - Si falta información, usa null o [].
    - Cada string ≤ 300 caracteres. Citas ≤ 120 (máx. 3). Extrae literalmente.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS (TODOS para “Sí”)
    1) Hay tema/propuesta en discusión.
    2) Aparecen frases breves con función de cierre (“es lo que hay”, “punto final”, etc.).
    3) La fuerza persuasiva principal es desalentar continuar el análisis.
    4) Falta o mínima presencia de justificación pertinente.

    DISTINCIONES
    - Slogan moviliza (no necesariamente cierra).
    - Whataboutism/Red herring desvían; el cliché termina la conversación.
    - Apelación a autoridad invoca autoridad; el cliché corta sin fundamento.
    - “Se acabó el tiempo” (logístico) no cuenta.

    SEÑALES
    - Imperativos de cierre; fatalismo/nihilismo; anti-intelectualismo; identitario expulsivo; relativismo simplista; mantras/hashtags.

    INSTRUCCIONES DE ANÁLISIS
    - Identifica el tema/propuesta en disputa (si existe).
    - Detecta clichés/frases hechas con función de cierre y clasifica el mecanismo primario.
    - Valora si desalientan la conversación (suppresses_discussion).
    - Evalúa presencia de razones (reasons_present).
    - Extrae hasta 3 citas literales (≤120) que evidencien el cierre.
    - Propón hasta 3 “reframes” que reabran el diálogo.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Conversation Killer” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Conversation Killer” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (texto literal):
    Our "unity in diversity" contrasts with the divisions everywhere else. Fin de la discussion

    Traza de decisión resumida (no imprimir en la salida):
    - Tema/propuesta detectado: afirmación sobre “unidad en la diversidad” vs. divisiones (hay tema → C1 ✓).
    - Clichés/imperativos: “unity in diversity” (cliché), “Fin de la discussion” (imperativo de cierre) → C2 ✓.
    - Mecanismo primario: imperativo_de_cierre (corta la conversación) → C3 ✓.
    - Razones presentes: bajas; no se aportan justificaciones sustantivas → C4 ✓.
    - Guardrails: citas literales ≤120 (máx. 3); salida SOLO JSON; strings ≤300.

    Salida esperada (solo JSON):
    {
      "es_conversation_killer": "Sí",
      "cliches_detected": [
        "unity in diversity",
        "Fin de la discussion"
      ],
      "quotes": [
        "unity in diversity",
        "Fin de la discussion"
      ],
      "primary_mechanism": "imperativo_de_cierre",
      "suppresses_discussion": {
        "value": true,
        "evidence": "La frase 'Fin de la discussion' cierra la conversación."
      },
      "reasons_present": {
        "level": "baja",
        "evidence": "La frase 'unity in diversity' es un cliché sin justificación adicional."
      },
      "topic_identified": "Unidad y diversidad en contraste con divisiones",
      "explanation": "El texto presenta un tema de unidad y diversidad, pero utiliza 'Fin de la discussion' para cerrar la conversación sin justificación.",
      "suggested_reframe": [
        "¿Cómo podemos mejorar nuestra unidad en diversidad?",
        "¿Qué ejemplos concretos de unidad en diversidad podemos seguir?",
        "¿Cómo se compara nuestra unidad con la de otros lugares?"
      ],
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: ConversationKillerJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )
    
class ConversationKillerRunner(TechniqueRunner):
    """Detects the CONVERSATION KILLER (Thought-Terminating Cliché) technique.

    Identifies whether persuasion relies on brief, formulaic phrases whose
    main function is to shut down further analysis rather than to provide
    substantive justification.

    ``postprocess`` returns a candidate dict with:
        model: ``"KILLER"``
        answer: ``"Sí"`` or ``"No"``
        confidence: float in [0, 1]
        span: up to 3 verbatim quotes or detected clichés
        rationale_summary: explanation including mechanism and reasons level
        raw: full ``ConversationKillerJudgment`` model dump
    """
    name = "KILLER"  # si prefieres "CONVERSATION_KILLER", cámbialo aquí
    signature = DetectaConversationKiller

    def postprocess(self, salida_obj: ConversationKillerJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.es_conversation_killer  # Sí/No

        # span: prioriza quotes, luego clichés detectados
        span = []
        if salida_obj.quotes:
            span = (salida_obj.quotes or [])[:3]
        elif salida_obj.cliches_detected:
            span = (salida_obj.cliches_detected or [])[:3]

        # rationale_summary compacto: explanation + mechanism + reasons level
        rationale_parts = [salida_obj.explanation.strip()]
        if salida_obj.primary_mechanism:
            rationale_parts.append(f"Mecanismo: {salida_obj.primary_mechanism}.")
        if salida_obj.reasons_present and salida_obj.reasons_present.level:
            rationale_parts.append(f"Razones: {salida_obj.reasons_present.level}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "primary_mechanism": salida_obj.primary_mechanism,
                "suppresses_discussion": bool(salida_obj.suppresses_discussion.value),
                "suppresses_discussion_evidence": salida_obj.suppresses_discussion.evidence,
                "reasons_level": salida_obj.reasons_present.level,
                "reasons_evidence": salida_obj.reasons_present.evidence,
                "topic_identified": salida_obj.topic_identified,
                "cliches_detected": salida_obj.cliches_detected,
                "suggested_reframe": salida_obj.suggested_reframe,
            },

            "raw": salida_obj.model_dump()
        }
    
# =========================================================
# Técnica SLIPPERY (Consequential Oversimplification)
# =========================================================

class BoolEvidence(BaseModel):
    """Boolean flag paired with an optional verbatim evidence excerpt.

    Attributes:
        value: The boolean verdict.
        evidence: Short verbatim quote (≤120 chars) supporting the verdict,
            or ``None`` if not available.
    """
    value: bool
    evidence: Optional[constr(max_length=120)] = None  # cita breve (≤120) o null


class SlipperySlopeJudgment(BaseModel):
    """LLM output for the SLIPPERY SLOPE (Consequential Oversimplification) detector.

    Key fields: ``is_slippery_slope`` (Sí/No verdict), ``initial_event``
    (the triggering event/proposal A), ``chain_steps`` (up to 3
    consequence steps B, C, …), ``end_point`` (extreme outcome),
    ``inevitability_claimed`` (claim of inevitability with evidence),
    ``support_provided`` (whether causal evidence is given),
    ``focus_on_chain_over_merits`` (whether the chain dominates over
    evaluating A), ``polarity`` (negative/positive/mixed), and
    ``confidence``.
    """
    is_slippery_slope: Literal["Sí", "No"]
    polarity: Optional[Literal["negativa", "positiva", "mixta"]] = None
    initial_event: Optional[constr(max_length=120)] = None
    chain_steps: List[constr(max_length=120)] = Field(default_factory=list)
    end_point: Optional[constr(max_length=120)] = None
    inevitability_claimed: BoolEvidence
    support_provided: BoolEvidence
    focus_on_chain_over_merits: BoolEvidence
    reason_short: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

    @field_validator("chain_steps")
    @classmethod
    def _limit_steps(cls, v: List[str]) -> List[str]:
        return v[:3]


class DetectaSlipperySlope(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Evalúa si el TEXTO usa “Consequential Oversimplification (Slippery Slope)”.

    GUARDRAIL (OBLIGATORIO)
    - Responde ÚNICAMENTE con un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si falta info, usa null o [].
    - Cada cadena ≤ 300 caracteres. Extrae como citas literales (≤120) cualquier evidencia textual.
    - No inventes ni reformules citas; máximo 3 citas totales repartidas entre las evidencias.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS ESTRICTOS (TODOS para “Sí”)
    1) Evento/propuesta inicial A identificable.
    2) Cadena de ≥2 consecuencias (B, C, …) que conduce a un desenlace extremo (end_point).
    3) Se insinúa/afirma inevitabilidad sin soporte suficiente, desplazando el análisis de los méritos de A.

    REGLAS DE DECISIÓN
    - “Sí” solo si se cumplen los 3 criterios.
    - Si existe soporte sustantivo suficiente para la cadena → “No”.
    - Si conviven promesa positiva y peligro negativo → polarity = “mixta”.
    - Con ambigüedad sustancial → “No” y completa con null/[] según corresponda.

    EXPLICACIÓN DE LA TÉCNICA (resumen)
    Forma típica: “si A ocurre, entonces B, C, D… ocurrirán”. Variante negativa: se busca RECHAZAR A con consecuencias negativas crecientes. Variante positiva (“escalera al paraíso”): se busca APOYAR A prometiendo beneficios crecientes. Rasgo distintivo: tratar la cadena como casi inevitable, minimizando mecanismos y evidencia.

    INSTRUCCIONES DE ANÁLISIS
    1) Normaliza y delimita: ignora mayúsculas/minúsculas; conserva las comillas del TEXTO para citas.
    2) Extrae A (evento/propuesta inicial) en ≤120.
    3) Identifica una cadena de al menos 2 consecuencias plausibles del discurso (B, C, …) en ≤120 cada una; limita a 3 pasos.
    4) Determina el end_point (desenlace extremo) en ≤120.
    5) Inevitabilidad: busca marcadores de certeza temporal o modal (p. ej., “will”, “inevitable”, “soon”, “sin duda”) y cita ≤120 si existe.
    6) Soporte: verifica si se ofrecen mecanismos, datos o analogías concretas. Si no, support_provided.value = false y evidence = null.
    7) Foco en cadena vs. méritos: si el énfasis principal está en el “domino” de consecuencias y no en discutir A, marca true y usa como evidencia una cita breve de la cadena (≤120).
    8) Polaridad: negativa/positiva según el signo del desenlace; “mixta” si coexisten.
    9) Compón la salida: respeta tipos/longitudes; no superes 3 citas en total entre los campos de evidencia; emite SOLO el JSON.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Slippery Slope” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Slippery Slope” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "If we allow same-sex marriage, we will soon see marriage between siblings or even marriages between humans and animals!"

    Traza de decisión resumida (no imprimir en la salida; sigue INSTRUCCIONES DE ANÁLISIS):
    - (2) A: "allow same-sex marriage".
    - (3) Cadena (≥2): "marriage between siblings" → "marriages between humans and animals".
    - (4) end_point: "marriages between humans and animals".
    - (5) Inevitabilidad (cita ≤120): "we will soon see".
    - (6) Soporte: no se ofrecen datos/mecanismos → false, evidence = null.
    - (7) Foco en cadena: enfatiza resultados extremos; evidencia (cita ≤120): "marriages between humans and animals".
    - (8) Polaridad: negativa.
    - (9) Salida SOLO JSON cumpliendo longitudes.

    Salida esperada (solo JSON):
    {
      "is_slippery_slope": "Sí",
      "polarity": "negativa",
      "initial_event": "allow same-sex marriage",
      "chain_steps": [
        "marriage between siblings",
        "marriages between humans and animals"
      ],
      "end_point": "marriages between humans and animals",
      "inevitability_claimed": {
        "value": true,
        "evidence": "we will soon see"
      },
      "support_provided": {
        "value": false,
        "evidence": null
      },
      "focus_on_chain_over_merits": {
        "value": true,
        "evidence": "marriages between humans and animals"
      },
      "reason_short": "El texto sostiene que permitir A conduce pronto a resultados extremos sin aportar mecanismos o evidencia suficiente, desplazando la discusión de los méritos de A.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    ─────────────────────────────────────────────────────────────────────────────

    SALIDA (JSON) exactamente con las claves pedidas.
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: SlipperySlopeJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )
    
class SlipperyRunner(TechniqueRunner):
    """Detects the SLIPPERY SLOPE (Consequential Oversimplification) technique.

    Identifies whether persuasion rests on an unjustified chain of
    consequences leading to an extreme outcome, presented as near-inevitable
    without sufficient causal support.

    ``postprocess`` returns a candidate dict with:
        model: ``"SLIPPERY"``
        answer: ``"Sí"`` or ``"No"``
        confidence: float in [0, 1]
        span: up to 3 verbatim evidences or chain/endpoint fragments
        rationale_summary: reason_short plus polarity annotation
        raw: full ``SlipperySlopeJudgment`` model dump
    """
    name = "SLIPPERY"  # coincide con tu candidates de ejemplo
    signature = DetectaSlipperySlope

    def postprocess(self, salida_obj: SlipperySlopeJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_slippery_slope  # Sí/No

        # span: prioriza evidencias literales; luego cadena/end_point
        span = []
        evidences = []
        for be in [salida_obj.inevitability_claimed,
                   salida_obj.focus_on_chain_over_merits,
                   salida_obj.support_provided]:
            if be and be.evidence:
                evidences.append(be.evidence)

        if evidences:
            # único y máx 3
            merged = []
            for e in evidences:
                if e and e not in merged:
                    merged.append(e)
            span = merged[:3]
        else:
            merged = []
            for x in [salida_obj.initial_event, *(salida_obj.chain_steps or []), salida_obj.end_point]:
                if x and x not in merged:
                    merged.append(x)
            span = merged[:3]

        # rationale_summary: reason_short + polaridad si existe
        rationale_parts = [salida_obj.reason_short.strip()]
        if salida_obj.polarity:
            rationale_parts.append(f"Polaridad: {salida_obj.polarity}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "polarity": salida_obj.polarity,
                "initial_event": salida_obj.initial_event,
                "chain_steps": salida_obj.chain_steps,
                "end_point": salida_obj.end_point,
                "inevitability_claimed": bool(salida_obj.inevitability_claimed.value),
                "inevitability_evidence": salida_obj.inevitability_claimed.evidence,
                "support_provided": bool(salida_obj.support_provided.value),
                "support_evidence": salida_obj.support_provided.evidence,
                "focus_on_chain_over_merits": bool(salida_obj.focus_on_chain_over_merits.value),
                "focus_evidence": salida_obj.focus_on_chain_over_merits.evidence,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica SLOGAN
# =========================================================

class SloganJudgment(BaseModel):
    """LLM judgment schema for the Slogan detection technique."""
    es_slogan: Literal["Sí", "No"]
    frases_detectadas: List[constr(max_length=120)] = Field(default_factory=list)
    objetivo_o_tema: Optional[constr(max_length=300)] = None
    rasgos_retóricos: List[
        Literal["rima","repetición","imperativo","hashtag","paralelismo","aliteración"]
    ] = Field(default_factory=list)
    apelaciones_emocionales: List[
        Literal["orgullo","miedo","ira","esperanza","identidad","seguridad","dignidad"]
    ] = Field(default_factory=list)
    etiquetado_o_estereotipos: List[constr(max_length=120)] = Field(default_factory=list)
    presencia_de_razones: Literal["ninguna", "mínima", "moderada", "sustantiva"]
    explicacion_breve: constr(max_length=300)
    ejemplos_de_texto: List[constr(max_length=120)] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaSlogan(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta uso de “Slogans”: frases breves, memorables y llamativas que
    apelan a emoción/identidad y sustituyen razones sustantivas.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si falta info, usa null o [].
    - Cada cadena ≤ 300 caracteres. Citas del TEXTO ≤ 120. No inventes ni reformules.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIO ESTRICTO (para “Sí”)
    - Hay consigna/leitmotiv/hashtag/imperativo breve y memorable.
    - La fuerza persuasiva principal es emocional/identitaria (≥50%).
    - Ausencia o mínima presencia de razones/evidencia pertinentes.

    DISTINCIONES / SEÑALES (resumen)
    - Name calling cuenta solo si aparece en forma de consigna y domina la persuasión.
    - Señales típicas: rima/aliteración/paralelismo/repetición; imperativos; hashtags (#...).
    - Ejemplos canónicos: “#NoMásX”, “Sí se puede”, “Ley y orden”.

    INSTRUCCIONES DE ANÁLISIS
    1) Preprocesa suavemente: minúsculas, minimiza puntuación; preserva frases tal cual para citas (≤120).
    2) Extrae candidatos de consigna: n-gramas breves (1–6 palabras), imperativos, hashtags, estructuras con paralelismo (“No X. No Y.”) o rima/aliteración.
    3) Señales retóricas: marca rima, aliteración, paralelismo, repetición, imperativo, hashtags cuando estén presentes.
    4) Evalúa fuerza persuasiva: ¿descansa principalmente en la consigna (emoción/identidad) o en razones verificables?
    5) Clasifica “presencia_de_razones” en {ninguna, mínima, moderada, sustantiva}.
    6) Construye salida JSON: cita literalmente 1–3 fragmentos breves; no reformules; respeta longitudes.
    7) Si dudas, prioriza precisión: “No” cuando la consigna no domine; “Sí” cuando cumpla todos los criterios.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Sloggans” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Sloggans” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "Immigrants welcome, racist not! No border. No control!"

    Traza de verificación resumida (no imprimir en la salida):
    - Normalización: minúsculas y puntuación mínima (mantener texto original para citas).
    - Candidatos de consigna detectados:
        • "Immigrants welcome, racist not!"  (breve, memorable; contraste binario)
        • "No border. No control!"           (paralelismo con “No …”)
    - Señales retóricas: paralelismo claro; puede percibirse rima/ritmo; eslóganes breves.
    - Apelaciones emocionales: identidad (“immigrants welcome”), dignidad (rechazo del “racist”).
    - Razones: no hay datos/evidencias; mensaje es consignístico.
    - Criterios estrictos:
        C1 (consigna breve/memorable): ✓
        C2 (persuasión emocional/identitaria ≥50%): ✓
        C3 (razones ausentes o mínimas): ✓
    - Guardrails: citas literales ≤120; salida SOLO JSON; campos ≤300.

    Salida esperada (solo JSON):
    {
      "es_slogan": "Sí",
      "frases_detectadas": [
        "Immigrants welcome, racist not!",
        "No border. No control!"
      ],
      "objetivo_o_tema": "Promover la aceptación de inmigrantes y rechazar el racismo y el control fronterizo.",
      "rasgos_retóricos": [
        "rima",
        "paralelismo"
      ],
      "apelaciones_emocionales": [
        "identidad",
        "dignidad"
      ],
      "etiquetado_o_estereotipos": [
        "racist"
      ],
      "presencia_de_razones": "ninguna",
      "explicacion_breve": "Las frases son consignas breves y memorables que apelan a la identidad y dignidad, sin ofrecer razones sustantivas.",
      "ejemplos_de_texto": [
        "Immigrants welcome, racist not!",
        "No border. No control!"
      ],
      "confidence": 0.95,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: SloganJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class SloganRunner(TechniqueRunner):
    """Detects the Slogan propaganda technique.

    A slogan is a brief, striking phrase repeated to reinforce a political or
    ideological position without substantive reasoning.

    Output dict keys (from postprocess):
        model: "SLOGAN"
        answer: "Sí" or "No"
        rationale_summary: brief explanation + presence of reasoning
        confidence: float in [0.0, 1.0]
        span: up to 3 literal text examples or detected phrases
        labels: frases_detectadas, objetivo_o_tema, rasgos_retóricos,
                apelaciones_emocionales, etiquetado_o_estereotipos,
                presencia_de_razones
        raw: full SloganJudgment model dump
    """
    name = "SLOGAN"
    signature = DetectaSlogan

    def postprocess(self, salida_obj: SloganJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.es_slogan  # Sí/No

        # span: prioriza ejemplos/frases detectadas
        span = []
        if salida_obj.ejemplos_de_texto:
            span = (salida_obj.ejemplos_de_texto or [])[:3]
        elif salida_obj.frases_detectadas:
            span = (salida_obj.frases_detectadas or [])[:3]

        # rationale_summary: explicación + presencia_de_razones (muy útil para tu juez)
        rationale_parts = [salida_obj.explicacion_breve.strip()]
        if salida_obj.presencia_de_razones:
            rationale_parts.append(f"Razones: {salida_obj.presencia_de_razones}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "frases_detectadas": salida_obj.frases_detectadas,
                "objetivo_o_tema": salida_obj.objetivo_o_tema,
                "rasgos_retóricos": salida_obj.rasgos_retóricos,
                "apelaciones_emocionales": salida_obj.apelaciones_emocionales,
                "etiquetado_o_estereotipos": salida_obj.etiquetado_o_estereotipos,
                "presencia_de_razones": salida_obj.presencia_de_razones,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica APPEAL_TO_VALUES
# =========================================================

ValueLiteral = Literal[
    "libertad","justicia","democracia","paz","transparencia","etica",
    "tradicion","dignidad","seguridad","merito","lealtad","orden","progreso","religión","creencias"
]

CooccurringLiteral = Literal[
    "Appeal to Authority", "Bandwagon", "Flag Waving", "Loaded Language"
]

class AppealToValuesJudgment(BaseModel):
    """LLM judgment schema for the Appeal to Values detection technique."""
    is_appeal_to_values: Literal["Sí", "No"]
    final_judgment: Literal["Sí, usa Appeal to Values", "No, no usa Appeal to Values"]
    claim_text: Optional[constr(max_length=300)] = None
    justification_text: Optional[constr(max_length=300)] = None
    values_invoked: List[ValueLiteral] = Field(default_factory=list)
    authority_framing_detected: bool
    loaded_language_only: bool
    evidence_balance: Literal["values_dominant", "balanced", "facts_dominant"]
    quotes_evidence: List[constr(max_length=120)] = Field(default_factory=list)
    co_occurring_techniques: List[CooccurringLiteral] = Field(default_factory=list)
    explanatory_note: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

    @field_validator("quotes_evidence")
    @classmethod
    def _limit_quotes(cls, v: List[str]) -> List[str]:
        return v[:3]


class DetectaAppealToValues(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Evalúa si el TEXTO usa “Appeal to Values”.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si faltan datos, usa null o [].
    - Cada cadena ≤ 300 caracteres. Citas ≤ 120 (máx. 3) y deben ser literales del TEXTO.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIO ESTRICTO (TODOS deben cumplirse para “Sí”)
    1) Hay ENUNCIADO (claim).
    2) Hay JUSTIFICACIÓN explícita.
    3) La justificación descansa principalmente (≥50%) en valores positivos abstractos (p. ej., justicia, libertad, tradición, religión/creencias).

    DISTINCIONES / SEÑALES
    - Loaded Language: uso de términos valorativos sin argumento (claim+justificación) → marcar loaded_language_only=true.
    - Appeal to Authority: se apoya en personas/entidades específicas (autoridad experta o institucional).
    - Bandwagon: “la mayoría/todos”; Flag Waving: orgullo/identidad grupal (“por la patria”, “nuestro pueblo”).
    - “facts_dominant”: datos, evidencia verificable, mecanismos causales detallados pesan más que los valores.
    - “balanced”: valores y evidencia sustantiva tienen peso similar.

    INSTRUCCIONES DE ANÁLISIS
    1) Normalizar mentalmente el TEXTO (minúsculas, reducir puntuación) solo para detección; las citas deben conservarse literales.
    2) Identificar claim (afirmación principal) y justificación (razón principal). Extraer citas exactas si existen.
    3) Mapear valores invocados hacia el conjunto permitido (libertad, justicia, democracia, paz, transparencia, etica,
       tradicion, dignidad, seguridad, merito, lealtad, orden, progreso, religión, creencias). Usar la forma normalizada.
    4) Evaluar el balance de evidencia:
       - values_dominant: la persuasión descansa ≥50% en apelaciones a valores abstractos.
       - balanced: peso comparable entre valores y evidencia sustantiva.
       - facts_dominant: predominan datos/mecanismos/verificación.
    5) Registrar co-ocurrencias si aparecen (Authority, Bandwagon, Flag Waving, Loaded Language).
    6) Limitar quotes_evidence a un máximo de 3 citas literales del TEXTO, cada una ≤120 caracteres.
    7) Completar el JSON conforme al esquema. No añadir nada fuera del JSON.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Appeal to Values” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Appeal to Values” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "We are against abortion because we prioritise the right to life of the unborn."

    Traza de decisión resumida (no imprimir en la salida):
    - Claim detectado: “We are against abortion”.
    - Justificación explícita: “because we prioritise the right to life of the unborn”.
    - Valor abstracto invocado: dignidad / derecho a la vida → normaliza a “dignidad”.
    - No hay autoridad citada, ni “todos/la mayoría”, ni apelación identitaria grupal.
    - El soporte persuasivo descansa en el valor abstracto (values_dominant).
    - Guardrails: 1 cita literal ≤120; salida SOLO JSON; strings ≤300.

    Salida esperada (solo JSON):
    {
      "is_appeal_to_values": "Sí",
      "final_judgment": "Sí, usa Appeal to Values",
      "claim_text": "We are against abortion",
      "justification_text": "because we prioritise the right to life of the unborn",
      "values_invoked": ["dignidad"],
      "authority_framing_detected": false,
      "loaded_language_only": false,
      "evidence_balance": "values_dominant",
      "quotes_evidence": ["because we prioritise the right to life of the unborn"],
      "co_occurring_techniques": [],
      "explanatory_note": "El argumento se basa en el valor abstracto de la dignidad y el derecho a la vida, sin evidencia factual.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: AppealToValuesJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class AppealToValuesRunner(TechniqueRunner):
    """Detects the Appeal to Values propaganda technique.

    Identifies arguments whose persuasive weight rests primarily (≥50%) on
    abstract positive values (justice, freedom, tradition, dignity, etc.)
    rather than factual evidence or substantive reasoning.

    Output dict keys (from postprocess):
        model: "VALUES"
        answer: "Sí" or "No"
        rationale_summary: explanatory_note + evidence_balance info
        confidence: float in [0.0, 1.0]
        span: up to 3 literal quotes or claim/justification excerpts
        labels: final_judgment, claim_text, justification_text, values_invoked,
                authority_framing_detected, loaded_language_only,
                evidence_balance, co_occurring_techniques
        raw: full AppealToValuesJudgment model dump
    """
    name = "VALUES"  # en tus ejemplos aparece como VALUES
    signature = DetectaAppealToValues

    def postprocess(self, salida_obj: AppealToValuesJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_appeal_to_values  # Sí/No

        # span: prioriza quotes_evidence, luego claim/justification, luego values_invoked
        span = []
        if salida_obj.quotes_evidence:
            span = (salida_obj.quotes_evidence or [])[:3]
        else:
            merged = []
            for x in [salida_obj.claim_text, salida_obj.justification_text]:
                if x and x not in merged:
                    merged.append(x)
            if not merged and salida_obj.values_invoked:
                merged.extend(list(salida_obj.values_invoked))
            span = merged[:3]

        # rationale_summary: explanatory_note + balance
        rationale_parts = [salida_obj.explanatory_note.strip()]
        if salida_obj.evidence_balance:
            rationale_parts.append(f"Balance: {salida_obj.evidence_balance}.")
        if salida_obj.loaded_language_only:
            rationale_parts.append("Loaded-language dominante (sin anclaje factual).")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "final_judgment": salida_obj.final_judgment,
                "claim_text": salida_obj.claim_text,
                "justification_text": salida_obj.justification_text,
                "values_invoked": salida_obj.values_invoked,
                "authority_framing_detected": bool(salida_obj.authority_framing_detected),
                "loaded_language_only": bool(salida_obj.loaded_language_only),
                "evidence_balance": salida_obj.evidence_balance,
                "co_occurring_techniques": salida_obj.co_occurring_techniques,
            },

            "raw": salida_obj.model_dump()
        }
        
# =========================================================
# Técnica RED_HERRING
# =========================================================

MechanismLiteral = Literal[
    "whataboutism","strawman","topic_shift","changing_goalposts",
    "ad_hominem_diversion","irrelevant_authority","anecdotal_diversion","other"
]

class FocusOnDistractor(BaseModel):
    """Whether the text's focus shifts to the distractor, with optional evidence."""
    value: bool
    evidence: Optional[constr(max_length=120)] = None  # cita breve o null

class RedHerringJudgment(BaseModel):
    """LLM judgment schema for the Red Herring detection technique."""
    is_red_herring: Literal["Sí", "No"]
    mechanism: MechanismLiteral
    original_claim_quote: Optional[constr(max_length=120)] = None
    original_topic: Optional[constr(max_length=300)] = None
    distractor_quote: Optional[constr(max_length=120)] = None
    distractor_topic: Optional[constr(max_length=300)] = None
    focus_on_distractor: FocusOnDistractor
    relevance_assessment: constr(max_length=300)
    indicators: List[constr(max_length=120)] = Field(default_factory=list)
    confounds_ruled_out: List[constr(max_length=120)] = Field(default_factory=list)
    final_note: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaRedHerring(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “Introducing Irrelevant Information (Red Herring)”.

    GUARDRAIL (OBLIGATORIO)
    - Responde ÚNICAMENTE con un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si faltan datos, usa null o [].
    - Citas ≤ 120 caracteres. Cada string ≤ 300 caracteres.
    - No inventes citas: extrae literalmente del TEXTO (máx. 3). Si no hay citas pertinentes, usa [].
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.
    

    CRITERIO ESTRICTO (TODOS)
    1) Hay tema/afirmación original que merece respuesta.
    2) Se introduce un tema distinto poco o no pertinente.
    3) La réplica se centra en el tema introducido, dejando el original sin abordar.

    SEÑALES (indicativas)
    - Whataboutism, cambio de tema explícito/implícito, autoridad/anécdota irrelevante,
      ad hominem de desvío, strawman (cuando desplaza el foco), mover la portería.

    DISTINCIONES (NO es Red Herring si…)
    - El ejemplo/autoridad conecta explícitamente con el tema original (pertinencia clara).
    - La respuesta multiaspecto aborda sustantivamente el tema original y el distractor es marginal.

    INSTRUCCIONES DE ANÁLISIS
    1) Extrae (cuando sea posible) una cita breve del tema/afirmación original y resume su tópico.
    2) Identifica el distractor: cita breve y tópico propuesto.
    3) Evalúa pertinencia: explica brevemente por qué el distractor es poco/no pertinente al original.
    4) Determina si el foco se desplaza al distractor (usa “focus_on_distractor.value” y una evidencia breve).
    5) Clasifica el mecanismo más adecuado (p. ej., "topic_shift", "whataboutism", etc.).
    6) Lista 1–3 “indicators” observables (p. ej., “shift in topic”, “irrelevant authority”).
    7) Enumera “confounds_ruled_out” si descartas explicaciones alternativas plausibles.
    8) Redacta “final_note” con una síntesis breve del veredicto.
    9) Verifica guardrails: límites de longitud, español, SOLO JSON, citas literales, hasta 3 citas.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “RedHerring” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “RedHerring” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "I have worked hard to help eliminate criminal activity. What we need is economic growth that can only come from the hands of leadership."

    Traza de decisión resumida (no imprimir en la salida):
    - Original claim/topic: Cita 1 = "I have worked hard to help eliminate criminal activity."
      → Tópico: esfuerzos contra la criminalidad.
    - Distractor: Cita 2 = "What we need is economic growth that can only come from the hands of leadership."
      → Tópico: crecimiento económico y liderazgo.
    - Pertinencia: el distractor no se vincula con el control del crimen ni responde a la afirmación original.
    - Desplazamiento de foco: la réplica se centra en crecimiento económico → “focus_on_distractor.value = true”,
      evidencia breve: el foco pasa a “economic growth” en lugar de la criminalidad.
    - Mecanismo: “topic_shift”.
    - Guardrails: citas ≤120; strings ≤300; español; SOLO JSON; máx. 3 citas.

    Salida esperada (solo JSON):
    {
      "is_red_herring": "Sí",
      "mechanism": "topic_shift",
      "original_claim_quote": "I have worked hard to help eliminate criminal activity.",
      "original_topic": "Efforts to eliminate criminal activity.",
      "distractor_quote": "What we need is economic growth that can only come from the hands of leadership.",
      "distractor_topic": "Economic growth and leadership.",
      "focus_on_distractor": {
        "value": true,
        "evidence": "The response shifts focus to economic growth instead of addressing criminal activity."
      },
      "relevance_assessment": "The introduction of economic growth as a topic diverts attention from the original claim about eliminating criminal activity.",
      "indicators": ["shift in topic", "introduction of unrelated topic"],
      "confounds_ruled_out": [],
      "final_note": "The response does not address the original claim about criminal activity, focusing instead on economic growth.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: RedHerringJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class RedHerringRunner(TechniqueRunner):
    """Detects the Red Herring (Introducing Irrelevant Information) propaganda technique.

    Identifies cases where an irrelevant topic is introduced to distract from
    the original claim, leaving it unaddressed. Common mechanisms include
    whataboutism, topic shifting, and ad hominem diversion.

    Output dict keys (from postprocess):
        model: "RED_HERRING"
        answer: "Sí" or "No"
        rationale_summary: relevance assessment + mechanism + final_note
        confidence: float in [0.0, 1.0]
        span: distractor_quote, then indicators, then original_claim_quote
        labels: mechanism, original_claim_quote, original_topic,
                distractor_quote, distractor_topic, focus_on_distractor,
                focus_on_distractor_evidence, indicators, confounds_ruled_out
        raw: full RedHerringJudgment model dump
    """
    name = "RED_HERRING"
    signature = DetectaRedHerring

    def postprocess(self, salida_obj: RedHerringJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_red_herring  # Sí/No

        # span: prioriza distractor_quote, luego indicators, luego original_claim_quote
        span = []
        if salida_obj.distractor_quote:
            span = [salida_obj.distractor_quote][:3]
        elif salida_obj.indicators:
            span = (salida_obj.indicators or [])[:3]
        elif salida_obj.original_claim_quote:
            span = [salida_obj.original_claim_quote][:3]

        # rationale_summary: relevance_assessment + mechanism + final_note (compacto)
        rationale_parts = [
            salida_obj.relevance_assessment.strip()
        ]
        if salida_obj.mechanism:
            rationale_parts.append(f"Mecanismo: {salida_obj.mechanism}.")
        if salida_obj.final_note:
            rationale_parts.append(salida_obj.final_note.strip())

        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "mechanism": salida_obj.mechanism,
                "original_claim_quote": salida_obj.original_claim_quote,
                "original_topic": salida_obj.original_topic,
                "distractor_quote": salida_obj.distractor_quote,
                "distractor_topic": salida_obj.distractor_topic,
                "focus_on_distractor": bool(salida_obj.focus_on_distractor.value),
                "focus_on_distractor_evidence": salida_obj.focus_on_distractor.evidence,
                "indicators": salida_obj.indicators,
                "confounds_ruled_out": salida_obj.confounds_ruled_out,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica STRAWMAN
# =========================================================

DistortionLiteral = Literal["caricatura","hombre de paja por extremos","selección sesgada"]

class EvidenceItem(BaseModel):
    """A labeled evidence quote with its role (original, strawman, or refutation)."""
    role: Literal["original","strawman","refutation"]
    quote: constr(max_length=120)

class StrawmanJudgment(BaseModel):
    """LLM judgment schema for the Strawman detection technique."""
    strawman_detected: Literal["Sí", "No"]
    original_position: Optional[constr(max_length=300)] = None
    strawman_position: Optional[constr(max_length=300)] = None
    refutation_excerpt: Optional[constr(max_length=120)] = None
    refutation_target: Literal["original", "strawman", "ambiguo"]
    distortion_types: List[DistortionLiteral] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    reasoning_brief: constr(max_length=300)
    severity: Literal["baja","media","alta"]
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaStrawman(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “Misrepresentation of Someone’s Position (Strawman)” en el TEXTO.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si faltan datos, usa null o [].
    - Todas las cadenas ≤300 caracteres. Citas ≤120 (extrae literalmente).
    - No incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIO ESTRICTO (TODOS deben cumplirse para “Sí”)
    1) Existe una posición original (explícita o atribuida).
    2) Aparece una versión reformulada que distorsiona/exagera/caricaturiza (strawman).
    3) La refutación se dirige principalmente (≥50%) a la versión distorsionada, no a la original.

    SEÑALES
    - “Lo que en realidad quieren…”, llevar a extremos (“todo/nada”, “poder absoluto”),
      caricaturas, atribuciones sin cita, selección sesgada de fragmentos.

    NO CONFUNDIR
    - Parafraseo fiel/simplificación benigna sin distorsión central.
    - Steelman (fortalecer la postura oponente).
    - Hombres de paja parciales sin refutación dirigida a la versión distorsionada.
    - Red herring/desvío de tema sin reformulación de la posición.
    - Hipérbole retórica general que no reorienta la refutación hacia la versión distorsionada.

    INSTRUCCIONES DE ANÁLISIS
    1) Extracción literal de evidencias:
       - Marca como "original" la cita que expresa/atribuye la postura inicial.
       - Marca como "strawman" la cita que la exagera/distorsiona.
       - Si hay refutación, extrae un breve fragmento como "refutation".
    2) Reconstrucción breve (≤300c):
       - Resume la posición original en "original_position".
       - Resume la versión distorsionada en "strawman_position".
    3) Objetivo de la refutación:
       - Determina si la refutación ataca a la original, al strawman, o es ambigua.
    4) Tipificar la distorsión:
       - Usa "caricatura" (exageración grotesca),
         "hombre de paja por extremos" (llevar a absolutos),
         "selección sesgada" (citas fuera de contexto).
       - Puedes combinar más de un tipo si aplica.
    5) Severidad:
       - "baja": distorsión ligera o local.
       - "media": distorsión clara que reorienta la discusión.
       - "alta": distorsión dominante que sustituye por completo la postura original.
    6) Consistencia y longitudes:
       - Verifica límites de longitud y literalidad de citas.
       - Emite SOLO el JSON conforme al esquema.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Strawman” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Strawman” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "the corporate (i.e. private sector) players in global governance are determined to have their agenda accepted everywhere — which is none other than to grant themselves full powers over the planet"

    Traza de decisión resumida (no imprimir en la salida):
    - Identificación de original: “...have their agenda accepted everywhere” enuncia una agenda, sin especificar su contenido.
    - Identificación de strawman: “...grant themselves full powers over the planet” extrema la agenda a control planetario.
    - Refutación: no hay fragmento que refute; el foco retórico se dirige a la versión extrema.
    - Tipificación: “caricatura” (exageración grotesca); también compatible con “hombre de paja por extremos”.
    - Severidad: “media” (la exageración redefine el sentido de “agenda”).
    - Validación de guardrails: citas literales ≤120, strings ≤300, salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "strawman_detected": "Sí",
      "original_position": "Corporate players in global governance have an agenda.",
      "strawman_position": "Their agenda is to grant themselves full powers over the planet.",
      "refutation_excerpt": null,
      "refutation_target": "strawman",
      "distortion_types": ["caricatura"],
      "evidence": [
        {
          "role": "original",
          "quote": "the corporate (i.e. private sector) players in global governance are determined to have their agenda accepted everywhere"
        },
        {
          "role": "strawman",
          "quote": "which is none other than to grant themselves full powers over the planet"
        }
      ],
      "reasoning_brief": "El texto exagera la 'agenda' corporativa hasta una caricatura de control planetario.",
      "severity": "media",
      "confidence": 0.8,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: StrawmanJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class StrawmanRunner(TechniqueRunner):
    """Detects the Strawman (Misrepresentation of Someone's Position) propaganda technique.

    Identifies cases where an opponent's position is distorted or caricatured
    so the refutation targets the distorted version rather than the original.

    Output dict keys (from postprocess):
        model: "STRAWMAN"
        answer: "Sí" or "No"
        rationale_summary: reasoning_brief + distortion_types + severity
        confidence: float in [0.0, 1.0]
        span: literal evidence quotes (original/strawman/refutation)
        labels: original_position, strawman_position, refutation_excerpt,
                refutation_target, distortion_types, severity, evidence
        raw: full StrawmanJudgment model dump
    """
    name = "STRAWMAN"
    signature = DetectaStrawman

    def postprocess(self, salida_obj: StrawmanJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.strawman_detected  # Sí/No

        # span: prioriza evidencia literal (original/strawman/refutation), luego posiciones
        span = []
        if salida_obj.evidence:
            merged = []
            for ev in salida_obj.evidence:
                q = getattr(ev, "quote", None)
                if q and q not in merged:
                    merged.append(q)
            span = merged[:3]
        else:
            merged = []
            for x in [salida_obj.original_position, salida_obj.strawman_position, salida_obj.refutation_excerpt]:
                if x and x not in merged:
                    merged.append(x)
            span = merged[:3]

        # rationale_summary: reasoning_brief + tipos + severidad
        rationale_parts = [salida_obj.reasoning_brief.strip()]
        if salida_obj.distortion_types:
            rationale_parts.append(f"Distorsión: {', '.join(salida_obj.distortion_types)}.")
        if salida_obj.severity:
            rationale_parts.append(f"Severidad: {salida_obj.severity}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "original_position": salida_obj.original_position,
                "strawman_position": salida_obj.strawman_position,
                "refutation_excerpt": salida_obj.refutation_excerpt,
                "refutation_target": salida_obj.refutation_target,
                "distortion_types": salida_obj.distortion_types,
                "severity": salida_obj.severity,
                "evidence": [ev.model_dump() for ev in (salida_obj.evidence or [])],
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica APPEAL_TO_FEAR_PREJUDICE
# =========================================================

TypeLiteral = Literal["miedo", "prejuicio", "ambos"]
StrengthLiteral = Literal["alta", "media", "baja"]

class AppealFearPrejudiceJudgment(BaseModel):
    """LLM judgment schema for the Appeal to Fear or Prejudice detection technique."""
    is_appeal_to_fear_prejudice: Literal["Sí", "No"]
    type: Optional[TypeLiteral] = None
    statement: Optional[constr(max_length=300)] = None
    justification_summary: Optional[constr(max_length=300)] = None
    targets: List[constr(max_length=300)] = Field(default_factory=list)
    fear_triggers: List[constr(max_length=300)] = Field(default_factory=list)
    loaded_language_terms: List[constr(max_length=300)] = Field(default_factory=list)
    evidence_quotes: List[constr(max_length=300)] = Field(default_factory=list)  # max 3 en prompt
    status_quo_as_alternative: bool
    co_occurs_consequential_oversimplification: bool
    co_occurs_false_dilemma: bool
    overall_strength: StrengthLiteral
    notes: Optional[constr(max_length=300)] = None
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaAppealToFearPrejudice(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “Appeal to Fear, Prejudice” (apelar al miedo y/o al prejuicio) en el TEXTO.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español. Si faltan datos, usa null o [].
    - Todas las cadenas ≤ 300 caracteres.
    - No inventes citas; extrae literalmente del TEXTO (máx. 3 en "evidence_quotes").
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIO ESTRICTO (para responder "Sí" deben cumplirse TODOS)
    1) Hay ENUNCIADO (claim) identificable.
    2) Hay JUSTIFICACIÓN explícita (aunque breve).
    3) La justificación descansa principalmente (≥50%) en inducir miedo/repulsión o prejuicio hacia un objetivo.

    SEÑALES (no exhaustivas)
    - Lenguaje alarmista o amenazas (“peligro”, “catástrofe”, “invasión”, “plaga”, “colapso”).
    - Estigmas, deshumanización, generalizaciones negativas hacia un grupo (“criminales”, “parásitos”, “terroristas”).
    - Escenarios temidos vagos/no proporcionados o desproporcionados respecto a la evidencia presentada.

    DISTINCIONES
    - Alerta basada en evidencia proporcional y concreta → No aplica.
    - Puede co-ocurrir con Consequential Oversimplification y False Dilemma.
    - Loaded Language puede aparecer, pero aquí el peso central es inducir miedo/prejuicio.

    INSTRUCCIONES DE ANÁLISIS
    - Normaliza levemente para detectar patrones (minúsculas, puntuación mínima) sin alterar el contenido citado.
    - Identifica el objetivo/“targets” (personas/grupos/ideas) y si la persuasión principal apela al miedo o al prejuicio:
        * type="miedo" → amenaza/temor es el vector persuasivo dominante.
        * type="prejuicio" → estigma generalizado o caracterización negativa esencialista.
        * type="ambos" → conviven miedo y estigma con peso comparable.
    - Distingue “fear_triggers” (escenarios/amenazas o marcos de temor, p. ej., “invasión”, “riesgo de atentado”) de “loaded_language_terms” (términos cargados puntuales, p. ej., “terroristas”, “plaga”).
      Si un disparador coincide con un término ya listado en “loaded_language_terms”, puedes dejar “fear_triggers” como [].
    - “status_quo_as_alternative” = true si se presenta explícitamente “mantener lo actual” como única/superior opción frente al miedo/prejuicio.
    - Marca co-ocurrencias:
        * co_occurs_consequential_oversimplification = true si se infla una cadena causal simplista (“si X entra → desastre”).
        * co_occurs_false_dilemma = true si se plantean solo 2 opciones excluyentes (“o prohibimos X o habrá caos”).
    - “overall_strength”:
        * alta: lenguaje intensamente alarmista/estigmatizante, sin matices, con foco persuasivo central en miedo/prejuicio.
        * media: presencia clara pero con contrapesos/ambigüedad parcial.
        * baja: señales débiles o subordinadas a razones sustantivas.
    - En “justification_summary”, explica brevemente por qué el soporte persuasivo principal es miedo/prejuicio.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Appeal to fear” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Appeal to fear” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "We must stop those refugees as they are terrorists."

    Traza de decisión resumida (no imprimir en la salida):
    - Detección de claim: “We must stop those refugees…”.
    - Justificación explícita: “…as they are terrorists.”
    - Peso persuasivo: recurre a un estigma fuerte (“terrorists”) aplicado a “refugees” → prevalece el prejuicio.
    - Objetivo/targets: “refugees”.
    - Loaded language: “terrorists”.
    - Triggers de miedo: pueden quedar vacíos si ya se captura el disparador como término cargado.
    - Sin evidencia proporcional ni matices → fuerza “alta”.
    - Guardrails: citas literales (≤3), cadenas ≤300, salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "is_appeal_to_fear_prejudice": "Sí",
      "type": "prejuicio",
      "statement": "We must stop those refugees as they are terrorists.",
      "justification_summary": "La frase atribuye a los refugiados ser 'terroristas', induciendo prejuicio contra ese grupo.",
      "targets": ["refugees"],
      "fear_triggers": [],
      "loaded_language_terms": ["terrorists"],
      "evidence_quotes": ["We must stop those refugees as they are terrorists."],
      "status_quo_as_alternative": false,
      "co_occurs_consequential_oversimplification": false,
      "co_occurs_false_dilemma": false,
      "overall_strength": "alta",
      "notes": null,
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: AppealFearPrejudiceJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )
    
class FearPrejudiceRunner(TechniqueRunner):
    """Detects the Appeal to Fear or Prejudice propaganda technique.

    Identifies arguments that exploit fear or group prejudice (e.g., targeting
    refugees, ethnic groups) using loaded language and fear triggers rather
    than substantive evidence.

    Output dict keys (from postprocess):
        model: "FEAR"
        answer: "Sí" or "No"
        rationale_summary: justification_summary + type + overall_strength
        confidence: float in [0.0, 1.0]
        span: evidence_quotes, then loaded_language_terms/fear_triggers
        labels: type, statement, targets, fear_triggers, loaded_language_terms,
                evidence_quotes, status_quo_as_alternative,
                co_occurs_consequential_oversimplification,
                co_occurs_false_dilemma, overall_strength, notes
        raw: full AppealFearPrejudiceJudgment model dump
    """
    name = "FEAR"  # coincide con tus candidates
    signature = DetectaAppealToFearPrejudice

    def postprocess(self, salida_obj: AppealFearPrejudiceJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_appeal_to_fear_prejudice  # Sí/No

        # span: prioriza evidence_quotes; luego loaded_language_terms/fear_triggers; luego statement
        span = []
        if salida_obj.evidence_quotes:
            span = (salida_obj.evidence_quotes or [])[:3]
        else:
            merged = []
            for x in (salida_obj.loaded_language_terms or []) + (salida_obj.fear_triggers or []):
                if x and x not in merged:
                    merged.append(x)
            if not merged and salida_obj.statement:
                merged.append(salida_obj.statement)
            span = merged[:3]

        # rationale_summary: justification_summary + tipo + fuerza
        base_reason = (salida_obj.justification_summary or "").strip()
        if not base_reason and salida_obj.notes:
            base_reason = salida_obj.notes.strip()

        rationale_parts = [base_reason] if base_reason else []
        if salida_obj.type:
            rationale_parts.append(f"Tipo: {salida_obj.type}.")
        if salida_obj.overall_strength:
            rationale_parts.append(f"Fuerza: {salida_obj.overall_strength}.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "type": salida_obj.type,
                "statement": salida_obj.statement,
                "targets": salida_obj.targets,
                "fear_triggers": salida_obj.fear_triggers,
                "loaded_language_terms": salida_obj.loaded_language_terms,
                "evidence_quotes": salida_obj.evidence_quotes,
                "status_quo_as_alternative": bool(salida_obj.status_quo_as_alternative),
                "co_occurs_consequential_oversimplification": bool(
                    salida_obj.co_occurs_consequential_oversimplification
                ),
                "co_occurs_false_dilemma": bool(salida_obj.co_occurs_false_dilemma),
                "overall_strength": salida_obj.overall_strength,
                "notes": salida_obj.notes,
            },

            "raw": salida_obj.model_dump()
        }
    
# =========================================================
# Técnica APPEAL_TO_AUTHORITY
# =========================================================

AuthorityValidity = Literal["valida", "invalida", "incierto"]
Dependency = Literal["principal", "secundaria", "no_aplica"]

class AppealToAuthorityJudgment(BaseModel):
    """LLM judgment schema for the Appeal to Authority detection technique."""
    is_appeal_to_authority: Literal["Sí", "No"]
    claim_detected: Optional[constr(max_length=300)] = None
    authority_mentioned: Optional[constr(max_length=300)] = None
    authority_evidence_spans: List[constr(max_length=200)] = Field(default_factory=list)
    authority_validity: AuthorityValidity
    dependency_on_authority: Dependency
    self_reference_as_authority: bool
    mere_sourcing: bool
    is_fallacious: bool
    justification: constr(max_length=400)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

    @field_validator("authority_evidence_spans")
    @classmethod
    def _limit_evidence(cls, v: List[str]) -> List[str]:
        return v[:3]


class DetectaAppealToAuthority(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta la técnica “Appeal to Authority” (apelar a la autoridad) y juzga si es falaz.

    DEFINICIÓN ESTRICTA
    (a) Existe afirmación/conclusión identificable.
    (b) La justificación descansa ≥50% en que “X lo dice/avala” (X como autoridad).

    SEÑALES
    “según…”, “de acuerdo con…”, “como afirma…”, “los expertos…”, “estudios demuestran…” (sin detalles),
    autoridad vaga o fuera de dominio, autorreferencia como experto, sustitución de datos/método por autoridad.

    DISTINCIONES
    - “Mero sourcing” (trazabilidad/cita de origen) ≠ técnica persuasiva por autoridad.
    - Ad populum (mayorías) ≠ autoridad.
    - Flag waving (identidad/grupo) ≠ autoridad.

    CRITERIOS ACOTADOS
    - authority_validity: valida / invalida / incierto (según pertinencia de dominio, credenciales y relevancia del experto para el claim).
    - dependency_on_authority: principal (soporte central) / secundaria (apoyo) / no_aplica.
    - is_fallacious = true si:
        • autoridad inválida o vaga, o
        • dependencia principal sin soporte sustantivo adicional
          EXCEPTO cuando la autoridad es claramente competente en el dominio y el claim coincide con consenso robusto del campo.
    - mere_sourcing = true si la mención es solo trazabilidad y el soporte no depende de la autoridad.
    - self_reference_as_authority = true si el hablante se cita a sí mismo como autoridad.

    INSTRUCCIONES DE ANÁLISIS (OBLIGATORIAS)
    1) Detecta la afirmación principal (“claim_detected”).
    2) Identifica menciones de autoridad (“authority_mentioned”) y extrae 1–3 evidencias literales (“authority_evidence_spans”, ≤200 c/u).
    3) Evalúa “authority_validity” por dominio, credenciales y pertinencia.
    4) Evalúa “dependency_on_authority” (¿la persuasión depende principalmente de la autoridad?).
    5) Distingue “mere_sourcing” (solo trazabilidad) de apelación persuasiva.
    6) Marca “self_reference_as_authority” si aplica.
    7) Decide “is_appeal_to_authority” y “is_fallacious” según los criterios acotados.
    8) Redacta “justification” en 2–4 frases (≤400), en español, sin razonamiento paso a paso.
    9) Devuelve “confidence” ∈ [0,1].
    10) Respeta límites de longitud; no inventes citas; usa null o [] si falta info.

    GUARDRAIL DE SALIDA
    - Devuelve ÚNICAMENTE un objeto JSON válido conforme al esquema. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Escribe en español.
    - Citas (authority_evidence_spans) ≤ 200 caracteres c/u, máx. 3. “justification” ≤ 400 caracteres.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Appeal to authority” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Appeal to authority” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (texto):
    "Richard Dawkins, an evolutionary biologist and perhaps the foremost expert in the field, says that evolution is true. Therefore, it’s true."

    Traza de decisión resumida (no imprimir en la salida):
    - Claim: “evolution is true”.
    - Autoridad detectada: “Richard Dawkins” (biólogo evolutivo; dominio pertinente).
    - Evidencia literal (1): oración que atribuye la verdad de la evolución a la autoridad de Dawkins.
    - Validity: “valida” (credenciales y dominio coinciden con el claim).
    - Dependencia: “principal” (la conclusión se apoya en que el experto lo dice).
    - Mero sourcing: no (la mención no es solo trazabilidad).
    - Falacia: no, porque la autoridad es competente y el claim coincide con consenso científico robusto.
    - Guardrails: todas las cadenas ≤ límites; salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "is_appeal_to_authority": "Sí",
      "claim_detected": "evolution is true",
      "authority_mentioned": "Richard Dawkins",
      "authority_evidence_spans": [
        "Richard Dawkins, an evolutionary biologist and perhaps the foremost expert in the field, says that evolution is true."
      ],
      "authority_validity": "valida",
      "dependency_on_authority": "principal",
      "self_reference_as_authority": false,
      "mere_sourcing": false,
      "is_fallacious": false,
      "justification": "El argumento se apoya en Richard Dawkins, experto reconocido en biología evolutiva, para sostener que la evolución es verdadera. La autoridad es pertinente y la dependencia es principal. Dado que el claim coincide con un consenso científico robusto, la apelación no resulta falaz.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: AppealToAuthorityJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class AuthorityRunner(TechniqueRunner):
    """Detects the Appeal to Authority propaganda technique.

    Identifies arguments where the persuasive weight rests primarily on an
    authority figure's endorsement rather than substantive evidence. Distinguishes
    between valid authority citation and fallacious appeal.

    Output dict keys (from postprocess):
        model: "AUTHORITY"
        answer: "Sí" or "No"
        rationale_summary: justification + mere_sourcing/fallacious flags
        confidence: float in [0.0, 1.0]
        span: authority_evidence_spans, then authority_mentioned/claim_detected
        labels: claim_detected, authority_mentioned, authority_evidence_spans,
                authority_validity, dependency_on_authority,
                self_reference_as_authority, mere_sourcing, is_fallacious
        raw: full AppealToAuthorityJudgment model dump
    """
    name = "AUTHORITY"   # coincide con tus candidates
    signature = DetectaAppealToAuthority

    def postprocess(self, salida_obj: AppealToAuthorityJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_appeal_to_authority  # Sí/No

        # span: prioriza evidence_spans; luego authority_mentioned/claim_detected
        span = []
        if salida_obj.authority_evidence_spans:
            span = (salida_obj.authority_evidence_spans or [])[:3]
        else:
            merged = []
            for x in [salida_obj.authority_mentioned, salida_obj.claim_detected]:
                if x and x not in merged:
                    merged.append(x)
            span = merged[:3]

        # rationale_summary: justification (ya viene compacta), + flags relevantes
        rationale_parts = [salida_obj.justification.strip()]
        if salida_obj.mere_sourcing:
            rationale_parts.append("Mención es solo trazabilidad (mere_sourcing=true).")
        if salida_obj.is_fallacious:
            rationale_parts.append("Apelación considerada falaz.")
        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "claim_detected": salida_obj.claim_detected,
                "authority_mentioned": salida_obj.authority_mentioned,
                "authority_evidence_spans": salida_obj.authority_evidence_spans,
                "authority_validity": salida_obj.authority_validity,
                "dependency_on_authority": salida_obj.dependency_on_authority,
                "self_reference_as_authority": bool(salida_obj.self_reference_as_authority),
                "mere_sourcing": bool(salida_obj.mere_sourcing),
                "is_fallacious": bool(salida_obj.is_fallacious),
            },

            "raw": salida_obj.model_dump()
        }
        
# =========================================================
# Técnica BANDWAGON
# =========================================================

BandwagonValue = Literal["Sí", "No"]
FallacyValue = Literal["Sí", "No", "Mixto", "No aplica"]
CriterionLiteral = Literal["≥50% popularidad", "relevancia válida", "evidencia sustantiva domina"]

class PressureToJoin(BaseModel):
    """Whether the text applies social pressure to join the majority."""
    value: bool
    evidence: Optional[constr(max_length=300)] = None

class OtherEvidence(BaseModel):
    """Whether substantive evidence other than popularity claims is present."""
    value: bool
    examples: List[constr(max_length=300)] = Field(default_factory=list)

class BandwagonDetected(BaseModel):
    """Detection result for the bandwagon appeal with supporting evidence."""
    value: BandwagonValue
    evidence: constr(max_length=300)

class FallaciousUse(BaseModel):
    """Whether the popularity appeal is used fallaciously, with criterion and evidence."""
    value: FallacyValue
    criterion: CriterionLiteral
    evidence: constr(max_length=300)

class BandwagonJudgment(BaseModel):
    """LLM judgment schema for the Bandwagon (Appeal to Popularity) detection technique."""
    is_bandwagon: Literal["Sí", "No"]
    bandwagon_detected: BandwagonDetected
    statement: Optional[constr(max_length=300)] = None
    popularity_claims: List[constr(max_length=300)] = Field(default_factory=list)
    referenced_group: List[constr(max_length=300)] = Field(default_factory=list)
    pressure_to_join: PressureToJoin
    other_substantive_evidence_present: OtherEvidence
    fallacious_use: FallaciousUse
    edge_case_notes: Optional[constr(max_length=300)] = None
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaBandwagon(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “Appeal to Popularity (Bandwagon)”.

    DEFINICIÓN ESTRICTA
    (a) Existe una afirmación o conclusión identificable.
    (b) La justificación descansa principalmente (≥50%) en alegatos de popularidad
        (“todos”, “la mayoría”, “consenso”, “70% dice…”, “nadie”, etc.).

    SEÑALES (indicativas, no exhaustivas)
    - Marcadores de popularidad: “todos”, “nadie”, “la mayoría”, “consenso”,
      “encuestas muestran…”, “X% apoya…”.
    - Presión a unirse: “no te quedes atrás”, “únete”, “serás el único que no…”.

    DIFERENCIAS / EXCEPCIONES (NO falaz típicamente)
    - Popularidad usada como contexto junto a evidencia sustantiva dominante
      (p. ej., datos metodológicamente sólidos, causalidad, comparaciones precisas).
    - “Consenso científico” con fundamento metodológico explícito y pertinente.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido en español. Sin texto fuera del JSON.
    - Si falta información, usa null o [] según corresponda.
    - Cada cadena ≤ 300 caracteres.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    INSTRUCCIONES DE ANÁLISIS
    1) Normaliza el TEXTO (minúsculas, puntuación mínima) sin alterar el contenido.
    2) Extrae la afirmación/conclusión (si existe) y guárdala en "statement" (≤300).
    3) Detecta y lista en "popularity_claims" las frases o cifras que aluden a
       popularidad (máx. necesario, cada una ≤300).
    4) Identifica el/los “referenced_group” (p. ej., “70%”, “la mayoría”).
    5) Evalúa “pressure_to_join”: ¿hay llamados a sumarse o evitar quedar fuera?
       - Si sí, value=true y sintetiza evidencia literal (≤300).
       - Si no, value=false y evidence=null.
    6) Evalúa “other_substantive_evidence_present”:
       - value=true si hay datos/razones sustantivas relevantes que dominan.
       - Añade 1–3 ejemplos textuales breves en "examples".
    7) Decide “is_bandwagon” y “bandwagon_detected” (value y evidence):
       - “Sí” si la justificación descansa ≥50% en popularidad.
       - “No” si la popularidad es secundaria o contexto.
    8) Decide “fallacious_use”:
       - value="Sí" y criterion="≥50% popularidad" si el peso persuasivo principal
         es la popularidad sin sostén sustantivo comparable.
       - value="No" y criterion="evidencia sustantiva domina" si la evidencia
         sustantiva es predominante.
       - "Mixto" si hay mezcla sin dominio claro; puedes usar
         criterion="relevancia válida".
       - "No aplica" en casos atípicos (p. ej., mera pregunta sin justificación).
    9) Usa "edge_case_notes" para aclarar ambigüedades (≤300) o deja null.
    10) Asigna "confidence" en [0.0, 1.0] según claridad de señales y consistencia.
    11) Verifica longitudes y emite SOLO el objeto JSON final.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Bandwagon” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Bandwagon” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "Would you vote for Putin as president? 70% say yes"

    Traza de decisión resumida (no imprimir en la salida):
    - (1) Afirmación/conclusión: implícita invitación a votar; la frase “70% say yes”
          funciona como justificación.
    - (2) Popularidad detectada: “70% say yes” → claim de mayoría.
    - (3) Grupo referido: “70%”.
    - (4) Presión a unirse: no se detecta lenguaje imperativo (“únete”, “no te quedes atrás”) → false.
    - (5) Evidencia sustantiva: no hay datos metodológicos ni razones independientes → value=false.
    - (6) Peso persuasivo: la justificación descansa ≥50% en popularidad.
    - (7) Decisiones:
          is_bandwagon="Sí";
          bandwagon_detected.value="Sí";
          fallacious_use.value="Sí";
          fallacious_use.criterion="≥50% popularidad";
          confidence≈0.9.
    - (8) Guardrails: todas las cadenas ≤300; salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "is_bandwagon": "Sí",
      "bandwagon_detected": {
        "value": "Sí",
        "evidence": "La afirmación de que el 70% dice que votaría por Putin sugiere un argumento basado en la popularidad."
      },
      "statement": "Would you vote for Putin as president? 70% say yes",
      "popularity_claims": ["70% say yes"],
      "referenced_group": ["70%"],
      "pressure_to_join": { "value": false, "evidence": null },
      "other_substantive_evidence_present": { "value": false, "examples": [] },
      "fallacious_use": {
        "value": "Sí",
        "criterion": "≥50% popularidad",
        "evidence": "La justificación descansa en que el 70% votaría por Putin, lo cual es un argumento de popularidad."
      },
      "edge_case_notes": null,
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: BandwagonJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class BandwagonRunner(TechniqueRunner):
    """Detects the Appeal to Popularity (Bandwagon) propaganda technique.

    Identifies arguments whose justification rests primarily (≥50%) on
    popularity claims ("everyone thinks", "70% say yes") rather than
    substantive reasoning or evidence.

    Output dict keys (from postprocess):
        model: "BANDWAGON"
        answer: "Sí" or "No"
        rationale_summary: bandwagon_detected evidence + fallacious_use info
        confidence: float in [0.0, 1.0]
        span: popularity_claims, then referenced_group, then statement
        labels: bandwagon_detected, statement, popularity_claims,
                referenced_group, pressure_to_join,
                other_substantive_evidence_present, fallacious_use,
                edge_case_notes
        raw: full BandwagonJudgment model dump
    """
    name = "BANDWAGON"
    signature = DetectaBandwagon

    def postprocess(self, salida_obj: BandwagonJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_bandwagon  # Sí/No

        # span: prioriza popularity_claims; luego referenced_group; luego statement/bandwagon evidence
        span = []
        if salida_obj.popularity_claims:
            span = (salida_obj.popularity_claims or [])[:3]
        elif salida_obj.referenced_group:
            span = (salida_obj.referenced_group or [])[:3]
        else:
            merged = []
            if salida_obj.statement:
                merged.append(salida_obj.statement)
            # evidencia corta del subcampo bandwagon_detected
            if salida_obj.bandwagon_detected and salida_obj.bandwagon_detected.evidence:
                merged.append(salida_obj.bandwagon_detected.evidence)
            span = merged[:3]

        # rationale_summary: bandwagon_detected.evidence + fallacious_use (compacto)
        rationale_parts = []
        if salida_obj.bandwagon_detected and salida_obj.bandwagon_detected.evidence:
            rationale_parts.append(salida_obj.bandwagon_detected.evidence.strip())

        if salida_obj.fallacious_use:
            fu = salida_obj.fallacious_use
            rationale_parts.append(
                f"Falacia: {fu.value} (criterio: {fu.criterion})."
            )

        # Si hay evidencia sustantiva dominante, dejamos nota corta
        if salida_obj.other_substantive_evidence_present and salida_obj.other_substantive_evidence_present.value:
            rationale_parts.append("Hay evidencia sustantiva relevante adicional.")

        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "bandwagon_detected": salida_obj.bandwagon_detected.model_dump()
                    if salida_obj.bandwagon_detected else None,
                "statement": salida_obj.statement,
                "popularity_claims": salida_obj.popularity_claims,
                "referenced_group": salida_obj.referenced_group,
                "pressure_to_join": salida_obj.pressure_to_join.model_dump()
                    if salida_obj.pressure_to_join else None,
                "other_substantive_evidence_present": salida_obj.other_substantive_evidence_present.model_dump()
                    if salida_obj.other_substantive_evidence_present else None,
                "fallacious_use": salida_obj.fallacious_use.model_dump()
                    if salida_obj.fallacious_use else None,
                "edge_case_notes": salida_obj.edge_case_notes,
            },

            "raw": salida_obj.model_dump()
        }
        
# =========================================================
# Técnica CASTING_DOUBT
# =========================================================

TacticType = Literal["antecedentes", "acciones_eventos", "afiliaciones", "motivos", "rasgos_personales"]

class TacticItem(BaseModel):
    """A single character-attack tactic with its type, quote, and explanation."""
    type: TacticType
    quote: constr(max_length=300)
    explanation: constr(max_length=300)

    @field_validator("quote")
    @classmethod
    def _limit_25_words(cls, v: str) -> str:
        words = v.split()
        if len(words) > 25:
            v = " ".join(words[:25])
        return v

class AddressesTopicSubstance(BaseModel):
    """Whether the argument addresses the substantive topic, with evidence."""
    value: bool
    evidence: constr(max_length=300)

class RelevanceAssessment(BaseModel):
    """Assessment of the relevance of character-based attacks to the topic."""
    value: Literal["no_relevante", "dudosa", "relevante"]
    justification: constr(max_length=300)

class CastingDoubtJudgment(BaseModel):
    """LLM judgment schema for the Casting Doubt detection technique."""
    is_casting_doubt: Literal["Sí", "No"]
    target: constr(max_length=300)
    topic_or_claim: constr(max_length=300)
    tactics_detected: List[TacticItem] = Field(default_factory=list)
    casting_doubt_claims: List[constr(max_length=300)] = Field(default_factory=list)
    addresses_topic_substance: AddressesTopicSubstance
    relevance_assessment: RelevanceAssessment
    weight_of_character_based_support: Literal["bajo", "medio", "alto"]
    strength: Literal["leve", "moderada", "fuerte"]
    reasoning_summary: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaCastingDoubt(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “Casting doubt”: desacreditar al objetivo (persona/grupo/entidad)
    en lugar de abordar el tema con razones/evidencia pertinentes.

    DEFINICIÓN (estricta)
    (a) Se centra en desacreditar por quién es/qué hizo/con quién se asocia/motivos/rasgos.
    (b) No aporta (o minimiza) evidencia o razones directamente sobre el asunto debatido.
    Excepciones (NO cuenta como casting doubt si DOMINAN): conflicto de interés directo y específico;
    credenciales estrictamente necesarias para la afirmación; historial metodológico relevante en el mismo dominio.

    REGLAS DE DECISIÓN
    - “Sí”: el soporte es mayoritariamente carácter-céntrico (≥50%) y la sustancia está ausente o es secundaria.
    - “No”: predomina evidencia/razones sobre el tema, o lo personal es pertinente y necesario.

    INSTRUCCIONES DE ANÁLISIS
    1) Normaliza el texto (minúsculas, puntuación mínima) sin alterar citas literales.
    2) Identifica el objetivo (target) y el tema/claim principal del texto.
    3) Extrae hasta 1–3 enunciados que funcionen como “casting_doubt_claims” (≤300 c/u).
    4) Detecta tácticas carácter-céntricas (TacticItem):
       - tipifica (antecedentes, acciones_eventos, afiliaciones, motivos, rasgos_personales),
       - incluye una “quote” literal ≤25 palabras,
       - explica en ≤300 por qué esa cita desacredita al objetivo en lugar de discutir la sustancia.
    5) Valora si se aborda la sustancia (addresses_topic_substance.value y evidence):
       datos, lógica, metodología, proporcionalidad, eficacia, etc.
    6) Evalúa pertinencia (relevance_assessment):
       “no_relevante”, “dudosa” o “relevante”, con justificación breve.
    7) Estima el peso carácter-céntrico (weight_of_character_based_support: bajo/medio/alto)
       y la fuerza retórica (strength: leve/moderada/fuerte).
    8) Emite juicio final (is_casting_doubt = “Sí”/“No”) y un resumen corto (reasoning_summary).
    9) GUARDRAILS DE FORMATO (OBLIGATORIO):
       - Devuelve ÚNICAMENTE un objeto JSON válido. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
       - Escribe en español.
       - Cadenas ≤300 caracteres; citas ≤25 palabras (aplica truncado si excede).
       - NO imprimas razonamiento paso a paso; piensa en privado y emite SOLO el JSON.
       - NO traduzcas al español el span. Manténlo en el idioma original.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Casting doubt” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Casting doubt” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "If you have nothing to hide, you have nothing to fear"

    Traza de decisión resumida (no imprimir en la salida):
    - Target: quienes cuestionan/rechazan la vigilancia (“you”).
    - Tema/claim: justificación de la vigilancia (si no escondes nada, no debes temer).
    - Tácticas:
      • “motivos”: imputa motivos ocultos (“If you have nothing to hide”).
      • “rasgos_personales”: asocia temor a falta de honestidad (“you have nothing to fear”).
    - Sustancia: no hay evidencia sobre eficacia, proporcionalidad o garantías → addresses_topic_substance.value = false.
    - Pertinencia: se desplaza el foco a motivos/rasgos del crítico → “no_relevante”.
    - Peso carácter-céntrico: alto; fuerza: fuerte.
    - Cumplimiento de guardrails: citas ≤25 palabras; salida SOLO JSON; campos ≤300 c.

    Salida esperada (solo JSON):
    {
      "is_casting_doubt": "Sí",
      "target": "Quienes cuestionan o rechazan la vigilancia/inspección (\"you\").",
      "topic_or_claim": "Justificación de la vigilancia: si no escondes nada, no debes temer.",
      "tactics_detected": [
        {
          "type": "motivos",
          "quote": "If you have nothing to hide",
          "explanation": "Imputa motivos ocultos al crítico: sugiere que quien objeta lo hace porque esconde algo."
        },
        {
          "type": "rasgos_personales",
          "quote": "you have nothing to fear",
          "explanation": "Asocia temor/objeción con sospecha o falta de honestidad del objetivo."
        }
      ],
      "casting_doubt_claims": [
        "If you have nothing to hide, you have nothing to fear."
      ],
      "addresses_topic_substance": {
        "value": false,
        "evidence": "No ofrece evidencia sobre eficacia, proporcionalidad o garantías de la vigilancia; es un eslogan sin sustancia."
      },
      "relevance_assessment": {
        "value": "no_relevante",
        "justification": "Traslada el foco a los motivos del crítico en vez de discutir la política de vigilancia."
      },
      "weight_of_character_based_support": "alto",
      "strength": "fuerte",
      "reasoning_summary": "El eslogan desacredita al crítico atribuyéndole motivos sospechosos y no aborda la política de vigilancia en sus méritos.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: CastingDoubtJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class CastingDoubtRunner(TechniqueRunner):
    """Detects the Casting Doubt propaganda technique.

    Identifies arguments that attack the credibility or character of a person
    or source (via background, actions, affiliations, motives, or personal
    traits) rather than addressing the substantive topic.

    Output dict keys (from postprocess):
        model: "CASTING_DOUBT"
        answer: "Sí" or "No"
        rationale_summary: reasoning_summary + weight + strength + relevance
        confidence: float in [0.0, 1.0]
        span: casting_doubt_claims, then tactic quotes, then target/topic
        labels: target, topic_or_claim, tactics_detected, casting_doubt_claims,
                addresses_topic_substance, relevance_assessment,
                weight_of_character_based_support, strength
        raw: full CastingDoubtJudgment model dump
    """
    name = "CASTING_DOUBT"
    signature = DetectaCastingDoubt

    def postprocess(self, salida_obj: CastingDoubtJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_casting_doubt  # Sí/No

        # span: prioriza casting_doubt_claims; luego quotes de tactics_detected; luego target/topic
        span = []
        if salida_obj.casting_doubt_claims:
            span = (salida_obj.casting_doubt_claims or [])[:3]
        elif salida_obj.tactics_detected:
            span = [t.quote for t in salida_obj.tactics_detected if t.quote][:3]
        else:
            merged = []
            for x in [salida_obj.target, salida_obj.topic_or_claim]:
                if x and x not in merged:
                    merged.append(x)
            span = merged[:3]

        # rationale_summary: reasoning_summary + peso/fuerza + relevancia
        rationale_parts = []
        if salida_obj.reasoning_summary:
            rationale_parts.append(salida_obj.reasoning_summary.strip())

        if salida_obj.weight_of_character_based_support:
            rationale_parts.append(f"Peso carácter-céntrico: {salida_obj.weight_of_character_based_support}.")

        if salida_obj.strength:
            rationale_parts.append(f"Fuerza: {salida_obj.strength}.")

        if salida_obj.relevance_assessment:
            ra = salida_obj.relevance_assessment
            rationale_parts.append(f"Pertinencia: {ra.value}.")

        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "target": salida_obj.target,
                "topic_or_claim": salida_obj.topic_or_claim,
                "tactics_detected": [t.model_dump() for t in (salida_obj.tactics_detected or [])],
                "casting_doubt_claims": salida_obj.casting_doubt_claims,
                "addresses_topic_substance": salida_obj.addresses_topic_substance.model_dump()
                    if salida_obj.addresses_topic_substance else None,
                "relevance_assessment": salida_obj.relevance_assessment.model_dump()
                    if salida_obj.relevance_assessment else None,
                "weight_of_character_based_support": salida_obj.weight_of_character_based_support,
                "strength": salida_obj.strength,
            },

            "raw": salida_obj.model_dump()
        }
        
# =========================================================
# Técnica FLAG_WAVING
# =========================================================

RelatedLiteral = Literal[
    "bandwagon", "glittering_generalities", "appeal_to_fear", "appeal_to_consequences"
]

class FlagWavingJudgment(BaseModel):
    """LLM judgment schema for the Flag Waving detection technique."""
    is_flag_waving: Literal["Sí", "No"]
    target_group: Optional[constr(max_length=300)] = None
    statement_detected: Optional[constr(max_length=300)] = None
    justification_group_based: Optional[constr(max_length=300)] = None
    other_substantive_reasons: List[constr(max_length=300)] = Field(default_factory=list)
    decision_rationale: constr(max_length=300)
    related_but_distinct: List[RelatedLiteral] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)


class DetectaFlagWaving(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “Flag Waving”: apelar al orgullo/identidad/beneficio de un grupo
    (p. ej., nación, partido, género, etnia, profesión) en lugar de razones sustantivas.

    INSTRUCCIONES DE ANÁLISIS
    1) Trabaja en español. Genera SOLO un objeto JSON válido conforme al esquema.
    2) Identifica:
       a) Enunciado/propuesta principal (p. ej., apoyar/rechazar X).
       b) Justificación basada en orgullo/identidad/beneficio de un grupo (“nuestro país”, “nuestra gente”, “para los trabajadores”, etc.).
    3) Dictamina “Sí” si ≥50% del peso persuasivo descansa en la apelación grupal (orgullo/identidad/beneficio para el grupo) y
       la evidencia sustantiva (datos/verificación/comparaciones precisas) está ausente o en segundo plano.
    4) Distingue de:
       - bandwagon: apelar a que “todos/la mayoría” apoyan algo.
       - glittering_generalities: términos positivos vagos sin marco explícito de grupo.
       - appeal_to_fear: apelar principalmente al miedo como razón.
       - appeal_to_consequences: enfatizar consecuencias (positivas/negativas) sin el marco “para nuestro grupo”.
    5) Excepción: si hay evidencia sustantiva dominante (datos, mecanismos causales explícitos, comparaciones precisas que sostienen la conclusión),
       NO clasifiques como Flag Waving.
    6) Guardrails: cadenas ≤ 300 caracteres; si falta info, usa null o []; salida SIN markdown, SIN comentarios, SOLO JSON.
       NO incluyas razonamiento paso a paso en la salida final (piensa en privado y emite únicamente el JSON). - NO traduzcas al español el span. Manténlo en el idioma original.

    GUARDRAIL (resumen)
    - Devuelve ÚNICAMENTE JSON válido con el esquema solicitado.
    - Si falta info, usa null o [].
    - Cada cadena ≤ 300 caracteres.
    - Español. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Flag waving” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Flag waving” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "We should make America great again, and restrict the immigration laws."

    Traza de decisión resumida (no imprimir en la salida):
    - Enunciado/propuesta: “restrict the immigration laws”.
    - Marco grupal: “make America great again” → apela al orgullo/identidad nacional (americanos).
    - Peso persuasivo: la justificación central es la identidad/beneficio del grupo (“América grande”), sin datos o mecanismos.
      → ≥50% basado en apelación grupal.
    - Diferenciación:
        * No es bandwagon: no invoca “todos/la mayoría”.
        * Puede contener un término loado (“great”), pero está anclado a identidad nacional → no es solo glittering_generalities.
        * No hay miedo explícito → no es appeal_to_fear.
        * Se menciona una consecuencia (restringir leyes) pero el soporte no es consecuencias evaluadas; es identidad grupal → no es appeal_to_consequences.
    - Guardrails: salida en español, JSON válido, campos ≤300c.

    Salida esperada (solo JSON):
    {
      "is_flag_waving": "Sí",
      "target_group": "Americanos",
      "statement_detected": "We should make America great again, and restrict the immigration laws.",
      "justification_group_based": "El enunciado apela al orgullo nacional de hacer a América grande nuevamente.",
      "other_substantive_reasons": [],
      "decision_rationale": "El argumento se apoya principalmente en la identidad y el orgullo nacional para respaldar la restricción de leyes de inmigración, sin ofrecer datos o mecanismos verificables. La apelación grupal domina sobre razones sustantivas.",
      "related_but_distinct": [],
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: FlagWavingJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class FlagWavingRunner(TechniqueRunner):
    """Detects the Flag Waving propaganda technique.

    Identifies arguments that appeal to group pride, identity, or benefit
    (nation, party, ethnicity, profession) as the primary justification,
    in the absence of substantive evidence.

    Output dict keys (from postprocess):
        model: "FLAG_WAVING"
        answer: "Sí" or "No"
        rationale_summary: decision_rationale + target_group + related techniques
        confidence: float in [0.0, 1.0]
        span: statement_detected, justification_group_based, target_group
        labels: target_group, statement_detected, justification_group_based,
                other_substantive_reasons, related_but_distinct
        raw: full FlagWavingJudgment model dump
    """
    name = "FLAG_WAVING"
    signature = DetectaFlagWaving

    def postprocess(self, salida_obj: FlagWavingJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_flag_waving  # Sí/No

        # span: prioriza statement_detected, luego justification_group_based, luego target_group
        span = []
        for x in [
            salida_obj.statement_detected,
            salida_obj.justification_group_based,
            salida_obj.target_group,
        ]:
            if x and x not in span:
                span.append(x)
        span = span[:3]

        # rationale_summary: decision_rationale + grupo/relacionados si aportan
        rationale_parts = []
        if salida_obj.decision_rationale:
            rationale_parts.append(salida_obj.decision_rationale.strip())

        if salida_obj.target_group:
            rationale_parts.append(f"Grupo objetivo: {salida_obj.target_group}.")

        if salida_obj.related_but_distinct:
            rationale_parts.append(
                "Relacionado pero distinto de: " + ", ".join(salida_obj.related_but_distinct) + "."
            )

        rationale_summary = " ".join(rationale_parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "target_group": salida_obj.target_group,
                "statement_detected": salida_obj.statement_detected,
                "justification_group_based": salida_obj.justification_group_based,
                "other_substantive_reasons": salida_obj.other_substantive_reasons,
                "related_but_distinct": salida_obj.related_but_distinct,
            },

            "raw": salida_obj.model_dump()
        }
        
# =========================================================
# Técnica SMEAR_POISONING (Questioning the Reputation / Poisoning the Well)
# =========================================================

ClaimType = Literal[
    "criminalidad","inmoralidad","corrupcion","deshonestidad",
    "extremismo","incompetencia","otro"
]
TopicRel = Literal["alta","media","baja","nula"]
EvidQual = Literal["ninguna","anecdotica","fuente_citada","datos"]

class ClaimItem(BaseModel):
    """A negative reputation claim with its type and severity rating (1–3)."""
    quote: constr(max_length=300)
    type: ClaimType
    severity_1to3: conint(ge=1, le=3)

class BoolWithEvidence(BaseModel):
    """A boolean flag paired with optional supporting evidence text."""
    value: bool
    evidence: Optional[constr(max_length=300)] = None

class SupportMix(BaseModel):
    """Breakdown of persuasive weight between reputation attacks and topic evidence (%)."""
    reputation_focus_pct: conint(ge=0, le=100)
    topic_evidence_pct: conint(ge=0, le=100)

class ExceptionsApplicable(BaseModel):
    """Whether any exceptions to the smear/poisoning rule apply, with reason."""
    value: bool
    reason: Optional[constr(max_length=300)] = None

class SmearJudgment(BaseModel):
    """LLM judgment schema for the Smear / Poisoning the Well detection technique."""
    final_judgment: Literal["Sí", "No"]
    target: constr(max_length=300)
    claims_detected: List[ClaimItem] = Field(default_factory=list)
    timing_preemptive: BoolWithEvidence
    topic_relevance: TopicRel
    support_mix: SupportMix
    evidence_quality: EvidQual
    imperatives_to_ignore_opponent: BoolWithEvidence
    exceptions_applicable: ExceptionsApplicable
    justification: constr(max_length=300)
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: constr(max_length=300)

class DetectaSmearPoisoning(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta la técnica “Questioning the Reputation / Poisoning the Well”:
    desacreditar a la persona/grupo/idea objetivo atacando reputación, carácter o moral, especialmente de modo
    preventivo, en lugar de discutir el tema de fondo.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve SOLO un objeto JSON válido que cumpla el esquema. Escribe en español.
    - No añadas texto fuera del JSON. No markdown. No comentarios.
    - No evalúes veracidad fáctica; solo la función argumentativa.
    - Límites de longitud: todas las strings ≤300 caracteres. Citas literales en "quote" ≤300.
    - Si falta información, usa null o [] según corresponda.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    INSTRUCCIONES DE ANÁLISIS
    1) Objetivo: identifica explícitamente la diana de la descalificación (persona/grupo/idea).
    2) Afirmaciones de reputación: extrae citas literales que ataquen carácter/moral/competencia y clasifícalas
       en ClaimType. Asigna severidad (1=mención leve, 2=acusación clara, 3=acusación grave/rotunda).
    3) Soporte vs. tema: estima la mezcla de soporte (SupportMix). Debe sumar 100:
       - reputation_focus_pct: % de peso persuasivo apoyado en ataques de reputación/carácter.
       - topic_evidence_pct: % apoyado en razones/evidencia directamente pertinentes al tema.
    4) Calidad de evidencia (EvidQual): "ninguna" (mero ataque), "anecdotica", "fuente_citada" (enlace/medio),
       "datos" (cifras/comparaciones verificables).
    5) Relevancia del ataque (TopicRel): "alta" si la reputación es intrínsecamente pertinente al asunto debatido,
       "nula" si no guarda relación sustantiva; usar "media"/"baja" en casos intermedios.
    6) Preempción (timing_preemptive): detecta lenguaje preventivo que busque sesgar a la audiencia antes de oír al oponente
       (p. ej., “no le crean”, “no la escuchen”). Cita evidencia breve si value=true.
    7) Imperativos para ignorar: detecta llamados a desoír al objetivo (imperatives_to_ignore_opponent).
    8) Excepciones (ExceptionsApplicable): marca true si el tema es precisamente la reputación (p. ej., idoneidad moral
       para un cargo) o existe conflicto de interés/credencial estrictamente relevante que justifique el foco.
    9) Dictamen final:
       - “Sí” si reputation_focus_pct ≥ 50 y el soporte temático es ausente/secundario, o si hay preempción clara,
         salvo que apliquen excepciones fuertes (p. ej., la evaluación ES sobre reputación).
       - “No” si el soporte principal es temático (topic_evidence_pct > reputation_focus_pct) o si el ataque a reputación
         es marginal/pertinente y no preemptivo.
    10) Justificación (≤300): explica brevemente por qué el dictamen es “Sí/No”, referenciando los factores clave
        (foco en reputación, preempción, relevancia, evidencia).

    CHECKLIST DE VALIDACIÓN (no imprimir en salida)
    - Mezcla SupportMix suma 100.
    - Citas literales en claims_detected reflejan el TEXTO y ≤300.
    - Campos booleanos con evidencia incluyen "evidence" si value=true (≤300).
    - coherence: final_judgment coherente con SupportMix, timing_preemptive y exceptions_applicable.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Smear” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Smear” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "My opponent has a record of lying and trying to cover her dishonest dealings with a pleasant smile. Don’t let her convince you to believe her words."

    Traza de decisión resumida (no imprimir en la salida):
    - Objetivo: "my opponent".
    - Afirmaciones de reputación: "a record of lying" (deshonestidad, severidad=3);
      "trying to cover her dishonest dealings with a pleasant smile" (deshonestidad, severidad=3).
    - Preempción: presente (“Don’t let her convince you…”).
    - Soporte vs. tema: 100% reputación, 0% evidencia del asunto debatido.
    - Evidencia: "ninguna" (no hay datos/fuentes).
    - Relevancia: "baja" (no se conecta con un tema sustantivo específico).
    - Excepciones: no aplican (el tema no es evaluar credenciales formales ni idoneidad).
    - Regla de decisión: reputation_focus_pct≥50 y preempción ⇒ dictamen “Sí”.

    Salida esperada (solo JSON):
    {
      "final_judgment": "Sí",
      "target": "my opponent",
      "claims_detected": [
        { "quote": "a record of lying", "type": "deshonestidad", "severity_1to3": 3 },
        { "quote": "trying to cover her dishonest dealings with a pleasant smile", "type": "deshonestidad", "severity_1to3": 3 }
      ],
      "timing_preemptive": { "value": true, "evidence": "Don’t let her convince you to believe her words." },
      "topic_relevance": "baja",
      "support_mix": { "reputation_focus_pct": 100, "topic_evidence_pct": 0 },
      "evidence_quality": "ninguna",
      "imperatives_to_ignore_opponent": { "value": true, "evidence": "Don’t let her convince you to believe her words." },
      "exceptions_applicable": { "value": false, "reason": null },
      "justification": "El texto se centra completamente en atacar la reputación del oponente, con preempción explícita y sin abordar el tema de fondo.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }

    """
    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: SmearJudgment = dspy.OutputField(desc="JSON válido según el esquema.")

        
class SmearPoisoningRunner(TechniqueRunner):
    """Detects the Smear/Poisoning the Well propaganda technique.

    Identifies arguments that attempt to discredit a person or source by
    making negative claims about their reputation (criminality, immorality,
    corruption, extremism) before or instead of addressing their arguments.

    Output dict keys (from postprocess):
        model: "SMEAR_POISONING"
        answer: "Sí" or "No" (derived from final_judgment)
        rationale_summary: justification + target + support_mix + key flags
        confidence: float in [0.0, 1.0]
        span: claims_detected quotes + preemptive/imperative evidence
        labels: target, claims_detected, timing_preemptive, topic_relevance,
                support_mix, evidence_quality, imperatives_to_ignore_opponent,
                exceptions_applicable
        raw: full SmearJudgment model dump
    """
    name = "SMEAR_POISONING"
    signature = DetectaSmearPoisoning

    def postprocess(self, salida_obj: SmearJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.final_judgment  # "Sí" / "No"

        # span: ponemos claims_detected quotes + evidencia preemptiva/imperativos si existen
        span = []
        for c in (salida_obj.claims_detected or []):
            if c.quote and c.quote not in span:
                span.append(c.quote)

        for be in [salida_obj.timing_preemptive, salida_obj.imperatives_to_ignore_opponent]:
            if be and be.value and be.evidence:
                if be.evidence not in span:
                    span.append(be.evidence)

        span = span[:3]

        # rationale_summary compacto
        parts = []
        if salida_obj.justification:
            parts.append(salida_obj.justification.strip())

        if salida_obj.target:
            parts.append(f"Objetivo: {salida_obj.target}.")

        # mezcla soporte
        if salida_obj.support_mix:
            parts.append(
                f"Soporte reputación/tema: "
                f"{salida_obj.support_mix.reputation_focus_pct}% / "
                f"{salida_obj.support_mix.topic_evidence_pct}%."
            )

        # flags clave
        if salida_obj.timing_preemptive and salida_obj.timing_preemptive.value:
            parts.append("Hay preempción preventiva.")
        if salida_obj.imperatives_to_ignore_opponent and salida_obj.imperatives_to_ignore_opponent.value:
            parts.append("Hay imperativos para ignorar al oponente.")
        if salida_obj.exceptions_applicable and salida_obj.exceptions_applicable.value:
            parts.append("Excepciones aplican.")

        rationale_summary = " ".join(parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "target": salida_obj.target,
                "claims_detected": [c.model_dump() for c in (salida_obj.claims_detected or [])],
                "timing_preemptive": salida_obj.timing_preemptive.model_dump() if salida_obj.timing_preemptive else None,
                "topic_relevance": salida_obj.topic_relevance,
                "support_mix": salida_obj.support_mix.model_dump() if salida_obj.support_mix else None,
                "evidence_quality": salida_obj.evidence_quality,
                "imperatives_to_ignore_opponent": (
                    salida_obj.imperatives_to_ignore_opponent.model_dump()
                    if salida_obj.imperatives_to_ignore_opponent else None
                ),
                "exceptions_applicable": (
                    salida_obj.exceptions_applicable.model_dump()
                    if salida_obj.exceptions_applicable else None
                ),
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica TU_QUOQUE
# =========================================================

Relevance = Literal["ninguna", "débil", "moderada", "fuerte"]

class ExceptionsApply(BaseModel):
    """Flags for legitimate exceptions to the Tu Quoque fallacy rule."""
    conflict_of_interest_or_deception: bool
    inconsistency_intrinsically_probative: bool
    substantive_evidence_present: bool

class TuQuoqueJudgment(BaseModel):
    """LLM judgment schema for the Tu Quoque (Appeal to Hypocrisy) detection technique."""
    final_judgment: Literal["Sí", "No"]
    target: Optional[constr(max_length=300)] = None
    accuser: Optional[constr(max_length=300)] = None
    issue_under_debate: Optional[constr(max_length=300)] = None
    hypocrisy_claims: List[constr(max_length=300)] = Field(default_factory=list)
    relevance_to_issue: Relevance
    primary_support_is_hypocrisy: bool
    exceptions_apply: ExceptionsApply
    justification: constr(max_length=400)   # 2–4 frases, máx. 400 c
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)

class DetectaTuQuoque(dspy.Signature):
    """
    ROL
    Analista de argumentos. Detecta “Appeal to Hypocrisy (Tu Quoque)”: desacreditar una posición
    resaltando hipocresía/inconsistencia, desplazando el fondo del asunto.

    INSTRUCCIONES DE ANÁLISIS
    - Paso 1 (en privado): identificar objetivo (target), acusador (accuser) y tema de fondo (issue_under_debate).
    - Paso 2 (en privado): extraer 1–3 afirmaciones de hipocresía/inconsistencia (hypocrisy_claims), citas o paráfrasis breves (≤300 c).
    - Paso 3 (en privado): evaluar la relevancia con respecto al tema (“ninguna”/“débil”/“moderada”/“fuerte”).
    - Paso 4 (en privado): decidir si el soporte principal (≥50%) es acusar de hipocresía → primary_support_is_hypocrisy.
    - Paso 5 (en privado): aplicar excepciones (exceptions_apply):
        • conflict_of_interest_or_deception: conflicto de interés específico o engaño relevante.
        • inconsistency_intrinsically_probative: la inconsistencia es, por sí misma, evidencia pertinente al tema.
        • substantive_evidence_present: hay razones/evidencia sustantivas (no meras descalificaciones).
    - Paso 6 (en privado): dictamen final (final_judgment):
        • “Sí” si (primary_support_is_hypocrisy == True) y no domina evidencia sustantiva; excepciones pueden mitigar.
        • “No” en caso contrario.
    - Paso 7: redactar justificación concisa (2–4 frases, ≤400 c) en español.
    - Importante: piensa en privado; NO imprimas razonamientos paso a paso. La salida debe ser SOLO el JSON.

    GUARDRAIL
    - Devuelve EXCLUSIVAMENTE un JSON válido conforme al esquema. En español.
    - Cadenas ≤ 300 c; justificación ≤ 400 c.
    - Si falta información, usa null o [] según corresponda.
    - No evalúes la veracidad fáctica; solo la función argumentativa.
    - No incluyas texto fuera del JSON, ni markdown, ni comentarios.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIO ESTRICTO DE “SÍ”
    - El soporte principal (≥50%) se basa en acusar hipocresía/inconsistencia del objetivo,
      y no se ofrecen (o quedan en segundo plano) razones/evidencia sustantivas sobre el tema.

    NO CONFUNDIR
    - Señalar conflictos de interés directos y específicos que sí son pertinentes al fondo.
    - Crítica metodológica pertinente al tema (eso NO es tu quoque).
    - Mera contradicción menor sin función persuasiva central.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “TuQuoque” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “TuQuoque” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "How can you demand that I eat less meat to reduce my carbon footprint if you yourself drive a big SUV and fly for holidays to Bali?"

    Traza de decisión resumida (no imprimir en la salida):
    - Identificación:
      • target: quien exige comer menos carne.
      • accuser: quien recibe la exigencia.
      • issue: reducir la huella de carbono comiendo menos carne.
    - Extracción de acusaciones de hipocresía: "you drive a big SUV", "you fly for holidays to Bali".
    - Relevancia: moderada (las supuestas incoherencias se relacionan con emisiones, pero no abordan directamente el argumento sobre carne).
    - Peso persuasivo: el ataque se centra en incoherencia del emisor (≥50%), sin evidencia sustantiva sobre la carne.
    - Excepciones:
      • conflict_of_interest_or_deception: False (no hay conflicto/engaño específico).
      • inconsistency_intrinsically_probative: False (la incoherencia no prueba por sí misma el punto sobre carne).
      • substantive_evidence_present: False (no hay datos/razones sustantivas).
    - Decisión: “Sí” (tu quoque). Redactar justificación breve en español. Salida SOLO JSON.

    Salida esperada (solo JSON):
    {
      "final_judgment": "Sí",
      "target": "la persona que exige comer menos carne",
      "accuser": "la persona a la que le piden comer menos carne",
      "issue_under_debate": "reducir la huella de carbono comiendo menos carne",
      "hypocrisy_claims": [
        "you drive a big SUV",
        "you fly for holidays to Bali"
      ],
      "relevance_to_issue": "moderada",
      "primary_support_is_hypocrisy": true,
      "exceptions_apply": {
        "conflict_of_interest_or_deception": false,
        "inconsistency_intrinsically_probative": false,
        "substantive_evidence_present": false
      },
      "justification": "El argumento desacredita la exigencia resaltando la supuesta hipocresía del emisor (SUV y vuelos), en lugar de aportar razones o evidencia sustantiva sobre el efecto de reducir la carne en la huella de carbono.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """

    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: TuQuoqueJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )


class TuQuoqueRunner(TechniqueRunner):
    """Detects the Tu Quoque (Appeal to Hypocrisy) propaganda technique.

    Identifies arguments that dismiss a claim by pointing out that the
    accuser behaves inconsistently (\"you do it too\"), deflecting rather
    than addressing the substantive issue.

    Output dict keys (from postprocess):
        model: "TU_QUOQUE"
        answer: "Sí" or "No" (derived from final_judgment)
        rationale_summary: justification + issue + target + relevance +
                           primary_support_is_hypocrisy + exceptions
        confidence: float in [0.0, 1.0]
        span: hypocrisy_claims (up to 3)
        labels: target, accuser, issue_under_debate, hypocrisy_claims,
                relevance_to_issue, primary_support_is_hypocrisy,
                exceptions_apply
        raw: full TuQuoqueJudgment model dump
    """
    name = "TU_QUOQUE"
    signature = DetectaTuQuoque

    def postprocess(self, salida_obj: TuQuoqueJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.final_judgment  # "Sí" / "No"

        # span: evidencia principal (claims de hipocresía)
        span = []
        for c in (salida_obj.hypocrisy_claims or []):
            if c and c not in span:
                span.append(c)
        span = span[:3]

        # rationale_summary compacto
        parts = []
        if salida_obj.justification:
            parts.append(salida_obj.justification.strip())

        if salida_obj.issue_under_debate:
            parts.append(f"Tema: {salida_obj.issue_under_debate}.")

        if salida_obj.target:
            parts.append(f"Objetivo: {salida_obj.target}.")

        if salida_obj.relevance_to_issue:
            parts.append(f"Relevancia al tema: {salida_obj.relevance_to_issue}.")

        parts.append(
            f"Soporte principal en hipocresía: "
            f"{'sí' if salida_obj.primary_support_is_hypocrisy else 'no'}."
        )

        ex = salida_obj.exceptions_apply
        if ex:
            ex_flags = []
            if ex.conflict_of_interest_or_deception:
                ex_flags.append("conflicto/engaño pertinente")
            if ex.inconsistency_intrinsically_probative:
                ex_flags.append("inconsistencia probativa")
            if ex.substantive_evidence_present:
                ex_flags.append("hay evidencia sustantiva")
            if ex_flags:
                parts.append("Excepciones: " + ", ".join(ex_flags) + ".")

        rationale_summary = " ".join(parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "target": salida_obj.target,
                "accuser": salida_obj.accuser,
                "issue_under_debate": salida_obj.issue_under_debate,
                "hypocrisy_claims": list(salida_obj.hypocrisy_claims or []),
                "relevance_to_issue": salida_obj.relevance_to_issue,
                "primary_support_is_hypocrisy": salida_obj.primary_support_is_hypocrisy,
                "exceptions_apply": (
                    salida_obj.exceptions_apply.model_dump()
                    if salida_obj.exceptions_apply else None
                ),
            },

            "raw": salida_obj.model_dump()
        }

    
from typing import List, Optional, Literal, Dict, Any, Type
from pydantic import BaseModel, Field, constr, confloat, field_validator
import pydantic
import dspy

# =========================================================
# Técnica GBA
# =========================================================

AssocType = Literal[
    "comparacion","pertenencia","apoyo","financiamiento",
    "coincidencia_doctrinal","proximidad","otra"
]

ArgRole = Literal[
    "sustituye_evidencia","complementa_evidencia","contexto_no_argumentativo"
]

class Association(BaseModel):
    """Description of the link between target and negative reference (type, explicitness, triggers)."""
    type: AssocType
    explicit: bool
    trigger_phrases: List[constr(max_length=300)] = Field(default_factory=list)

class ExceptionsCheck(BaseModel):
    """Flags for legitimate exceptions to the Guilt by Association fallacy."""
    relevance_specificity: bool
    independent_evidence_present: bool
    notes: Optional[constr(max_length=300)] = None

class GBAJudgment(BaseModel):
    """LLM judgment schema for the Guilt by Association detection technique."""
    is_guilt_by_association: Literal["Sí", "No"]
    target: constr(max_length=300)
    negative_reference: constr(max_length=300)
    association: Association
    argumentative_role: ArgRole
    evidence_quotes: List[constr(max_length=300)] = Field(default_factory=list)  # 1–3 fragmentos, ≤20 palabras c/u
    why_50w_max: constr(max_length=500)   # se validará a ≤50 palabras
    exceptions_check: ExceptionsCheck
    borderline: bool
    detected_language: constr(max_length=10)
    notes: Optional[constr(max_length=300)] = None
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)

    # Guardrails: evidencias (≤3) y cada una ≤20 palabras
    @field_validator("evidence_quotes")
    @classmethod
    def _limit_quotes(cls, v: List[str]) -> List[str]:
        out = []
        for s in v[:3]:
            words = s.split()
            if len(words) > 20:
                s = " ".join(words[:20])
            out.append(s)
        return out

    # Guardrail: justificar en ≤50 palabras
    @field_validator("why_50w_max")
    @classmethod
    def _limit_50_words(cls, v: str) -> str:
        words = v.split()
        if len(words) > 50:
            v = " ".join(words[:50])
        return v

    # Redondear confianza a 2 decimales
    @field_validator("confidence")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        return round(float(v), 2)


class DetectaGuiltByAssociation(dspy.Signature):
    """
    ROL
    Analista de argumentos. Detecta “Guilt by Association (Reductio ad Hitlerum)”.

    SALIDA
    Devuelve SOLO un JSON válido que cumple exactamente el esquema GBAJudgment (en español).

    INSTRUCCIONES DE ANÁLISIS (aplícalas en privado; NO imprimir pasos)
    1) Idioma: detecta el idioma dominante del TEXTO (ej.: "es", "en") y devuélvelo en "detected_language".
    2) Normaliza: minúsculas, puntuación mínima, sin alterar sentido; conserva citas literales para "evidence_quotes".
    3) Objetivo y referente:
       - target = persona/grupo/idea/actividad que se descalifica.
       - negative_reference = entidad con connotación fuertemente negativa para audiencia general.
    4) Tipo de vínculo (association.type) y explicitud:
       - comparacion/analogía: "como X", "igual que X".
       - pertenencia/alianza: "son del mismo bando", "aliados de".
       - apoyo/financiamiento: "respaldados/financiados por".
       - coincidencia_doctrinal: "misma ideología que".
       - proximidad: "marcharon junto a", "comparten escenario".
       - otra: especifica en "trigger_phrases".
       - explicit = True si el vínculo está nombrado; False si sólo insinuado.
    5) Rol argumentativo (argumentative_role):
       - sustituye_evidencia: la asociación es la razón principal para rechazar/atacar.
       - complementa_evidencia: asociación secundaria junto a pruebas sustantivas.
       - contexto_no_argumentativo: mención contextual sin función persuasiva.
    6) Excepciones (exceptions_check):
       - relevance_specificity = True si hay mecanismos/políticas/citas concretas y pertinentes.
       - independent_evidence_present = True si hay evidencia independiente más allá del vínculo.
       - Si ambas son True y no hay conclusión descalificadora basada en la asociación, probablemente NO es GBA.
    7) Evidencias:
       - Extrae 1–3 citas literales (≤20 palabras c/u) que muestren la asociación y/o el salto a descalificar.
    8) Veredicto y justificación:
       - is_guilt_by_association = "Sí" si la fuerza persuasiva depende mayoritariamente del vínculo negativo.
       - why_50w_max: justificación clara, neutral, ≤50 palabras.
       - borderline = True si el caso es limítrofe/ambiguo.
    9) Notas:
       - Si hay múltiples objetivos/referentes, prioriza el más saliente y detalla el resto en "notes".
    10) Guardrails estrictos:
       - SOLO JSON. Sin markdown, sin comentarios, sin pasos internos.
       - Mantén todas las longitudes especificadas y el idioma en español (salvo las citas textuales).
       - NO traduzcas al español el span. Manténlo en el idioma original.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “GuiltByAssociation” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “GuiltByAssociation” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). NO imprimir el checklist en la salida.
    TEXTO (entrada):
    "Only one kind of person can think in that way: a communist."

    Checklist de decisión (no imprimir):
    - Idioma detectado: "en".
    - Objetivo (target): "personas que piensan de esa manera".
    - Referente negativo (negative_reference): "comunista".
    - Vínculo: coincidencia_doctrinal; explicit = True. Disparadores: "Only one kind of person", "a communist".
    - Rol argumentativo: sustituye_evidencia (la asociación descalifica sin pruebas independientes).
    - Excepciones:
        relevance_specificity = False (no hay mecanismos/políticas concretas).
        independent_evidence_present = False (no hay evidencias adicionales).
    - Evidencias (≤20 palabras): seleccionar la oración clave que realiza la asociación.
    - Veredicto: "Sí". Justificación ≤50 palabras.
    - Borderline: False. Confianza ~0.90.

    Salida esperada (SOLO JSON):
    {
      "is_guilt_by_association": "Sí",
      "target": "personas que piensan de esa manera",
      "negative_reference": "comunista",
      "association": {
        "type": "coincidencia_doctrinal",
        "explicit": true,
        "trigger_phrases": [
          "Only one kind of person",
          "a communist"
        ]
      },
      "argumentative_role": "sustituye_evidencia",
      "evidence_quotes": [
        "Only one kind of person can think in that way: a communist"
      ],
      "why_50w_max": "Se descalifica una forma de pensar vinculándola a 'comunista' sin aportar evidencias independientes; la asociación cumple la función persuasiva principal.",
      "exceptions_check": {
        "relevance_specificity": false,
        "independent_evidence_present": false,
        "notes": null
      },
      "borderline": false,
      "detected_language": "en",
      "notes": null,
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """

    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: GBAJudgment = dspy.OutputField(desc="JSON válido conforme al esquema.")
    
class GuiltByAssociationRunner(TechniqueRunner):
    """Detects the Guilt by Association propaganda technique.

    Identifies arguments that discredit a target by linking them to a
    negative reference (person, group, or ideology), implying shared
    negative qualities without independent evidence.

    Output dict keys (from postprocess):
        model: "GUILT_BY_ASSOCIATION"
        answer: "Sí" or "No"
        rationale_summary: why_50w_max + target + negative_reference +
                           association type + argumentative_role + exceptions
        confidence: float in [0.0, 1.0]
        span: evidence_quotes (up to 3)
        labels: target, negative_reference, association, argumentative_role,
                exceptions_check, borderline
        raw: full GBAJudgment model dump
    """
    name = "GUILT_BY_ASSOCIATION"
    signature = DetectaGuiltByAssociation

    def postprocess(self, salida_obj: GBAJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_guilt_by_association  # "Sí" / "No"

        # span: usamos evidence_quotes como evidencia principal
        span = []
        for q in (salida_obj.evidence_quotes or []):
            if q and q not in span:
                span.append(q)
        span = span[:3]

        # rationale_summary compacto
        parts = []
        if salida_obj.why_50w_max:
            parts.append(salida_obj.why_50w_max.strip())

        if salida_obj.target:
            parts.append(f"Objetivo: {salida_obj.target}.")

        if salida_obj.negative_reference:
            parts.append(f"Referente negativo: {salida_obj.negative_reference}.")

        assoc = salida_obj.association
        if assoc:
            parts.append(
                f"Vínculo: {assoc.type}, "
                f"{'explícito' if assoc.explicit else 'insinuado'}."
            )

        if salida_obj.argumentative_role:
            parts.append(f"Rol argumentativo: {salida_obj.argumentative_role}.")

        ex = salida_obj.exceptions_check
        if ex:
            ex_bits = []
            ex_bits.append(
                f"especificidad pertinente: {'sí' if ex.relevance_specificity else 'no'}"
            )
            ex_bits.append(
                f"evidencia independiente: {'sí' if ex.independent_evidence_present else 'no'}"
            )
            parts.append("Excepciones: " + ", ".join(ex_bits) + ".")

        if salida_obj.borderline:
            parts.append("Caso limítrofe.")

        rationale_summary = " ".join(parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "target": salida_obj.target,
                "negative_reference": salida_obj.negative_reference,
                "association": (
                    salida_obj.association.model_dump()
                    if salida_obj.association else None
                ),
                "argumentative_role": salida_obj.argumentative_role,
                "exceptions_check": (
                    salida_obj.exceptions_check.model_dump()
                    if salida_obj.exceptions_check else None
                ),
                "borderline": salida_obj.borderline,
                "detected_language": salida_obj.detected_language,
            },

            "raw": salida_obj.model_dump()
        }


# =========================================================
# Técnica NAME CALLING
# =========================================================    
    
PolarityLit = Literal["peyorativa","laudatoria","mixta"]
TypeLit = Literal[
    "ideológica/partidaria","insulto","dehumanizante",
    "moralizante","asociativa","miedo/odio","adoración/halo"
]
TargetKindLit = Literal["individuo","grupo","ideología"]
StanceLit = Literal["endoso","reporte/cita","irónico/ambiguo"]
ArgStructLit = Literal[
    "solo etiquetamiento",
    "argumento con etiquetamiento",
    "argumento sin etiquetamiento"
]

class LabelTarget(BaseModel):
    """The target of a name-calling label (kind and span text)."""
    kind: TargetKindLit
    span: constr(max_length=300)

class LabelItem(BaseModel):
    """A detected name-calling label with polarity, type, target, stance, and evidence."""
    span: constr(max_length=300)
    polarity: PolarityLit
    types: List[TypeLit] = Field(default_factory=list)
    target: LabelTarget
    stance: StanceLit
    evidence_window: constr(max_length=300)
    reason: constr(max_length=300)

class ManipulativeWording(BaseModel):
    """Whether the text uses manipulative wording beyond basic name-calling."""
    value: bool
    evidence: Optional[constr(max_length=300)] = None

class NameCallingJudgment(BaseModel):
    """LLM judgment schema for the Name Calling / Labeling detection technique."""
    uses_name_calling: Literal["Sí", "No"]
    severity: conint(ge=0, le=3)
    labels_detected: List[LabelItem] = Field(default_factory=list)
    manipulative_wording: ManipulativeWording
    argument_structure: ArgStructLit
    notes: Optional[constr(max_length=300)] = None
    justification: constr(max_length=400)   # 2–4 frases, ≤400 c
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)

class DetectaNameCalling(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta “name calling / labelling”: etiquetas cargadas dirigidas a una persona/grupo
    que sustituyen razones por caracterizaciones esencialistas.

    ENTRADA
    TEXTO: <texto tal cual, entre comillas si aplica>

    GUARDRAILS (OBLIGATORIOS)
    - Responde EXCLUSIVAMENTE con un objeto JSON válido conforme al esquema. Sin texto fuera del JSON. Sin markdown. Sin comentarios.
    - Trabaja en español. Cada cadena ≤ 300 caracteres (salvo "justification" ≤ 400).
    - Mantén la clasificación estrictamente basada en el TEXTO dado. No inventes contenido.
    - No reveles cadenas de pensamiento ni pasos internos; piensa en privado y emite solo el JSON final.
    - "evidence_window" debe ser el fragmento CONTIGUO mínimo que contenga la etiqueta y señales de su uso (≤300 c).
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS ESTRICTOS
    - Identifica términos etiquetantes (sustantivos/adjetivos cargados) dirigidos a un objetivo explícito (persona/grupo/ideología).
    - Clasifica polaridad (peyorativa/laudatoria/mixta) y tipo(s) (insulto, dehumanizante, etc.).
    - Determina la postura (stance): endoso, reporte/cita (neutral o ajena), o irónico/ambiguo.
    - Evalúa si la etiqueta sustituye razones (premisas→conclusión) y si apela a emociones/identidades.
    - Reduce falsos positivos: definiciones, citas neutrales extensas, análisis académico descriptivo, usos técnicos no cargados.

    SEVERIDAD (0–3)
    - 0: no hay name calling o es marginal; 1: accesorio; 2: central pero no exclusivo; 3: eje dominante del argumento.
    - "argument_structure":
      * "solo etiquetamiento": ≥50% del peso persuasivo recae en etiquetas cargadas, sin razones sustantivas.
      * "argumento con etiquetamiento": hay razones, pero el etiquetamiento aporta peso significativo.
      * "argumento sin etiquetamiento": no hay etiquetas cargadas con función persuasiva.

    INSTRUCCIONES DE ANÁLISIS
    1) Normaliza mínimamente (minúsculas, puntuación básica) sin perder citas textuales.
    2) Extrae candidatos a etiqueta (nombres/adjetivos valorativos); descarta términos funcionales o meramente descriptivos.
    3) Vincula cada etiqueta con su objetivo explícito (target) y determina stance.
    4) Asigna polarity y type(s) (pueden coexistir varios tipos).
    5) Elige "evidence_window" como el span contiguo mínimo que soporte la clasificación.
    6) Decide "argument_structure" y "severity" según el peso persuasivo relativo del etiquetamiento vs. razones.
    7) "manipulative_wording": true si hay lenguaje diseñado para dirigir la valoración (hipérboles, adjetivos cargados, epítetos); incluye evidencia breve.
    8) Redacta "justification" en 2–4 frases, concisa y fundada en el texto.
    9) Si no hay etiquetas válidas, usa: uses_name_calling="No", severity=0, listas vacías, manipulative_wording.value=false.

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “NameCalling” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “NameCalling” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "My opponent is a flip-flop man who cannot make up his mind. He changes mind with the breeze! How could anyone follow such a weak-willed flip-flopper?"

    Traza de decisión resumida (no imprimir en la salida):
    - Detección de etiquetas: "flip-flop man", "weak-willed flip-flopper" → términos peyorativos dirigidos a un individuo (oponente).
    - Target: "My opponent" explícito; stance = endoso (el hablante adopta la descalificación).
    - Tipificación: "insulto" (caracterización negativa del carácter/consistencia).
    - Evidence windows: spans mínimos que contienen la etiqueta y su contexto inmediato.
    - Estructura/Severidad: el texto se apoya principalmente en epítetos sin razones sustantivas → "solo etiquetamiento", severidad=2 (central, casi dominante).
    - Manipulative wording: presente (adjetivos y epítetos que dirigen la valoración).

    Salida esperada (solo JSON):
    {
      "uses_name_calling": "Sí",
      "severity": 2,
      "labels_detected": [
        {
          "span": "flip-flop man",
          "polarity": "peyorativa",
          "types": ["insulto"],
          "target": { "kind": "individuo", "span": "My opponent" },
          "stance": "endoso",
          "evidence_window": "My opponent is a flip-flop man who cannot make up his mind.",
          "reason": "The term 'flip-flop man' is used to discredit the opponent by implying inconsistency and unreliability."
        },
        {
          "span": "weak-willed flip-flopper",
          "polarity": "peyorativa",
          "types": ["insulto"],
          "target": { "kind": "individuo", "span": "My opponent" },
          "stance": "endoso",
          "evidence_window": "How could anyone follow such a weak-willed flip-flopper?",
          "reason": "The term 'weak-willed flip-flopper' is used to portray the opponent as indecisive and lacking strength of character."
        }
      ],
      "manipulative_wording": {
        "value": true,
        "evidence": "The use of 'flip-flop man' and 'weak-willed flip-flopper' aims to manipulate the audience's perception of the opponent."
      },
      "argument_structure": "solo etiquetamiento",
      "notes": null,
      "justification": "The text uses derogatory labels to attack the opponent's character without providing substantive arguments, relying on emotional appeal.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """

    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: NameCallingJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class NameCallingRunner(TechniqueRunner):
    """Detects the Name Calling / Labeling propaganda technique.

    Identifies the use of derogatory labels, slurs, or pejorative terms
    to attack or discredit a target, with the label functioning as the
    primary persuasive element rather than substantive argument.

    Output dict keys (from postprocess):
        model: "NAME_CALLING"
        answer: "Sí" or "No"
        rationale_summary: justification + severity + argument_structure +
                           label examples + manipulative_wording flag
        confidence: float in [0.0, 1.0]
        span: evidence_window excerpts from detected labels (up to 3)
        labels: severity, argument_structure, labels_detected,
                manipulative_wording
        raw: full NameCallingJudgment model dump
    """
    name = "NAME_CALLING"
    signature = DetectaNameCalling

    def postprocess(self, salida_obj: NameCallingJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.uses_name_calling  # "Sí" / "No"

        # span principal: ventanas de evidencia contiguas (hasta 3)
        span: List[str] = []
        for item in (salida_obj.labels_detected or []):
            ew = (item.evidence_window or "").strip()
            if ew and ew not in span:
                span.append(ew)
        span = span[:3]

        # resumen compacto
        parts: List[str] = []
        if salida_obj.justification:
            parts.append(salida_obj.justification.strip())

        parts.append(f"Severidad: {salida_obj.severity}.")
        parts.append(f"Estructura: {salida_obj.argument_structure}.")

        if salida_obj.labels_detected:
            n_labels = len(salida_obj.labels_detected)
            parts.append(f"Etiquetas detectadas: {n_labels}.")
            # mostrar una lista corta de spans y polaridad/tipos
            brief_labels = []
            for li in salida_obj.labels_detected[:3]:
                tps = ",".join(li.types) if li.types else "sin_tipo"
                brief_labels.append(f"'{li.span}' ({li.polarity}; {tps})")
            parts.append("Ejemplos: " + "; ".join(brief_labels) + ".")

        mw = salida_obj.manipulative_wording
        if mw:
            parts.append(
                "Lenguaje manipulativo: "
                + ("sí" if mw.value else "no")
                + (f" ({mw.evidence})" if mw.value and mw.evidence else "")
                + "."
            )

        if salida_obj.notes:
            parts.append(f"Notas: {salida_obj.notes.strip()}")

        rationale_summary = " ".join(parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "severity": salida_obj.severity,
                "argument_structure": salida_obj.argument_structure,
                "labels_detected": [
                    li.model_dump() for li in (salida_obj.labels_detected or [])
                ],
                "manipulative_wording": (
                    salida_obj.manipulative_wording.model_dump()
                    if salida_obj.manipulative_wording else None
                ),
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica CAUSAL OVERSIMPLIFICATION
# =========================================================  

class CausalOversimplificationJudgment(BaseModel):
    """LLM judgment schema for the Causal Oversimplification detection technique."""
    uses_co: Literal["Sí", "No"]
    identification_of_causes: List[constr(max_length=300)] = Field(default_factory=list)
    identification_of_effect: Optional[constr(max_length=300)] = None
    review_of_alternative_factors: List[constr(max_length=300)] = Field(default_factory=list)

    # Ejemplos:
    # "Solo relación causa–efecto lineal" | "Considera múltiples interacciones"
    causal_interactions: constr(max_length=300)

    # Ejemplos:
    # "Demasiado simplista" | "Complejidad adecuada"
    evaluation_of_complexity: constr(max_length=120)

    # 2–4 frases, máx. 400 caracteres
    justification: constr(max_length=400)

    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)

    # Guardrail suave: limitar alternativas a máx. 6 (como indicas)
    @field_validator("review_of_alternative_factors")
    @classmethod
    def _limit_alternatives(cls, v: List[str]) -> List[str]:
        return v[:6]

    # Redondear confianza a 2 decimales (consistencia)
    @field_validator("confidence")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        return round(float(v), 2)

class DetectaCausalOversimplification(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Detecta si el TEXTO comete “causal oversimplification”:
    atribuir un fenómeno complejo a una sola causa, sin considerar factores múltiples e interrelacionados.

    FORMATO DE SALIDA (SOLO JSON; sin markdown, sin texto extra)
    Debes completar ÚNICAMENTE estas claves:
      - uses_co: "Sí" | "No"
      - identification_of_causes: [str]
      - identification_of_effect: str|null
      - review_of_alternative_factors: [str]
      - causal_interactions: str
      - evaluation_of_complexity: str
      - justification: str (2–4 frases, ≤400 caracteres)
      - confidence: float en [0.0, 1.0]
      - justificacion_confidence: str (2–4 frases, ≤400 caracteres)

    GUARDRAILS (OBLIGATORIO)
    - Devuelve SOLO un objeto JSON válido. Nada fuera del JSON.
    - Escribe en español (frases concisas, neutrales y factuales).
    - Si falta info, usa null o [] según corresponda.
    - No inventes contenido que no esté justificado por el texto o por factores plausibles del mundo real.
    - No incluyas razonamiento paso a paso en la salida; piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    INSTRUCCIONES DE ANÁLISIS
    1) Localiza explícitamente las causas mencionadas en el TEXTO (términos o frases causales).
    2) Identifica el efecto principal que el TEXTO intenta explicar o justificar.
    3) Enumera 3–6 factores alternativos plausibles (del mundo real) que podrían explicar el efecto pero no se mencionan.
    4) Determina si el TEXTO considera interacciones entre múltiples factores o si ofrece un vínculo lineal causa→efecto.
    5) Evalúa si la explicación es demasiado simple dada la complejidad esperable del fenómeno.
    6) Emite el juicio final en uses_co y redacta una justificación breve (2–4 frases) que cite los elementos clave.
    7) Mantén consistencia terminológica con las claves pedidas y respeta el límite de longitud.

    NO CONFUNDIR CON
    - Críticas válidas con múltiples razones sustantivas.
    - Narrativas causales con varios factores y mecanismos explícitos.
    - Referencias históricas usadas como contexto (no como causa única invalidante).

    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “Causal Oversimplification” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “Causal Oversimplification” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.

    Entrada (TEXTO):
    "A nation deflects criticism of its recent human rights violations by pointing to the history of slavery in the United States"

    Traza de verificación resumida (no imprimir en la salida):
    - Paso 1 (Causas): Se identifica una sola “causa” invocada: la historia de la esclavitud en EE. UU.
    - Paso 2 (Efecto): Desviar/invalidar la crítica actual sobre violaciones de DD. HH.
    - Paso 3 (Alternativas plausibles no mencionadas): decisiones recientes del gobierno, marco legal y su aplicación,
      contexto político, condiciones socioeconómicas, estándares internacionales, responsabilidad institucional.
    - Paso 4 (Interacciones): El TEXTO no plantea interacciones; usa una referencia histórica única como justificación lineal.
    - Paso 5 (Complejidad): Para evaluar violaciones de DD. HH. se requieren múltiples factores y mecanismos;
      reducirlo a una sola referencia histórica es simplista.
    - Paso 6 (Guardrails): Salida SOLO JSON, en español, longitudes respetadas.

    Salida esperada (solo JSON):
    {
      "uses_co": "Sí",
      "identification_of_causes": [
        "La historia de la esclavitud en Estados Unidos (usada como explicación única para desestimar la crítica actual)"
      ],
      "identification_of_effect": "Desviar o invalidar la crítica a las recientes violaciones de derechos humanos del país.",
      "review_of_alternative_factors": [
        "Decisiones y prácticas recientes del gobierno y fuerzas de seguridad",
        "Marco legal interno y su aplicación selectiva",
        "Contexto político (autoritarismo, represión, corrupción)",
        "Condiciones socioeconómicas y conflictos internos",
        "Presiones y estándares internacionales de derechos humanos",
        "Responsabilidad institucional y mecanismos de rendición de cuentas"
      ],
      "causal_interactions": "Solo se considera una relación causa–efecto lineal (la referencia histórica basta para invalidar la crítica).",
      "evaluation_of_complexity": "Demasiado simplista",
      "justification": "El texto usa una referencia histórica única para rechazar la crítica presente, ignorando múltiples factores relevantes y sus interacciones. Reduce un fenómeno complejo (violaciones y su evaluación) a una sola causa externa, sin ofrecer análisis de mecanismos ni contexto, lo que constituye causal oversimplification.",
      "confidence": 0.82,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """

    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: CausalOversimplificationJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class CausalOversimplificationRunner(TechniqueRunner):
    """Detects the Causal Oversimplification propaganda technique.

    Identifies arguments that attribute a complex outcome to a single or
    very few causes, ignoring relevant alternative factors and causal
    interactions.

    Output dict keys (from postprocess):
        model: "CAUSAL_OVERSIMPLIFICATION"
        answer: "Sí" or "No"
        rationale_summary: justification + cited causes + effect +
                           omitted factors + causal_interactions + complexity
        confidence: float in [0.0, 1.0]
        span: empty list (no literal quotes in schema)
        labels: identification_of_causes, identification_of_effect,
                review_of_alternative_factors, causal_interactions,
                evaluation_of_complexity
        raw: full CausalOversimplificationJudgment model dump
    """
    name = "CAUSAL_OVERSIMPLIFICATION"
    signature = DetectaCausalOversimplification

    def postprocess(self, salida_obj: CausalOversimplificationJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.uses_co  # "Sí" / "No"

        # No hay citas literales en el esquema → span vacío por defecto.
        span: List[str] = []

        parts: List[str] = []
        if salida_obj.justification:
            parts.append(salida_obj.justification.strip())

        if salida_obj.identification_of_causes:
            parts.append("Causas citadas: " + "; ".join(salida_obj.identification_of_causes[:3]) + ".")

        if salida_obj.identification_of_effect:
            parts.append(f"Efecto: {salida_obj.identification_of_effect}.")

        if salida_obj.review_of_alternative_factors:
            parts.append(
                "Factores alternativos omitidos: "
                + "; ".join(salida_obj.review_of_alternative_factors[:4])
                + "."
            )

        parts.append(f"Interacciones causales: {salida_obj.causal_interactions}.")
        parts.append(f"Complejidad: {salida_obj.evaluation_of_complexity}.")

        rationale_summary = " ".join(parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "identification_of_causes": salida_obj.identification_of_causes,
                "identification_of_effect": salida_obj.identification_of_effect,
                "review_of_alternative_factors": salida_obj.review_of_alternative_factors,
                "causal_interactions": salida_obj.causal_interactions,
                "evaluation_of_complexity": salida_obj.evaluation_of_complexity,
            },

            "raw": salida_obj.model_dump()
        }

# =========================================================
# Técnica FLASE DILEMMA
# =========================================================      

class Exclusivity(BaseModel):
    """Whether the dilemma is presented as mutually exclusive, with evidence."""
    value: bool
    evidence: Optional[constr(max_length=300)] = None

    # Guardrail suave: evidencia ≤120 si existe (coherente con tu prompt)
    @field_validator("evidence")
    @classmethod
    def _limit_evidence_120(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v[:120]

class ImperativeClosure(BaseModel):
    """Whether the text forces a choice with imperative language, with evidence."""
    value: bool
    evidence: Optional[constr(max_length=300)] = None

    @field_validator("evidence")
    @classmethod
    def _limit_evidence_120(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v[:120]

class FalseDilemmaJudgment(BaseModel):
    """LLM judgment schema for the False Dilemma detection technique."""
    is_fd: Literal["Sí", "No"]
    options_extracted: List[constr(max_length=300)] = Field(default_factory=list)
    exclusivity_claimed: Exclusivity
    third_options_possible: List[constr(max_length=300)] = Field(default_factory=list)
    imperative_closure: ImperativeClosure
    erases_continuum: bool
    justification: constr(max_length=400)   # 2–4 frases, máx. 400
    confidence: confloat(ge=0.0, le=1.0)
    justificacion_confidence: pydantic.constr(max_length=300)

    # Guardrail suave: limitar terceras opciones a máx. 6
    @field_validator("third_options_possible")
    @classmethod
    def _limit_third_options(cls, v: List[str]) -> List[str]:
        return v[:6]

    # Redondeo a 2 decimales como en otros módulos
    @field_validator("confidence")
    @classmethod
    def _two_decimals(cls, v: float) -> float:
        return round(float(v), 2)

class DetectaFalsoDilema(dspy.Signature):
    """
    ROL
    Eres un analista de argumentos. Determina si el texto comete la falacia “falso dilema”.

    ENTRADA
    TEXTO: cadena tal cual aparece.

    GUARDRAIL (OBLIGATORIO)
    - Devuelve ÚNICAMENTE un objeto JSON válido que cumpla el esquema.
    - Escribe en español.
    - Si falta info, usa null o [] donde corresponda.
    - Cada cadena ≤ 300 caracteres; la justificación ≤ 400. Citas ≤ 120.
    - No inventes ni reformules citas: extrae literalmente del TEXTO cuando cites.
    - NO incluyas razonamiento paso a paso en la salida. Piensa en privado y emite SOLO el JSON.
    - NO traduzcas al español el span. Manténlo en el idioma original.

    CRITERIOS ESTRICTOS PARA “Sí”
    1) Se presentan dos opciones (o un conjunto estrecho) y se sugiere/afirma exclusividad o exhaustividad.
    2) Existen terceras opciones plausibles o un continuo intermedio omitido/ignorado.
    3) (Señales reforzantes, no estrictamente necesarias) Llamados imperativos que cierran alternativas; borrado explícito del continuo.

    INSTRUCCIONES DE ANÁLISIS
    - Paso A: Extrae literalmente las opciones propuestas por el texto (lista corta).
    - Paso B: Detecta marcadores de exclusividad/exhaustividad (p. ej., “o… o…”, “solo”, “no hay otra opción”); cita un fragmento breve como evidencia si lo hay.
    - Paso C: Enumera 2–6 terceras opciones plausibles (políticas intermedias, enfoques mixtos, etc.).
    - Paso D: Identifica si hay “cierre imperativo” que descarte alternativas (p. ej., “por eso no apoyaremos…”); cita un fragmento breve si aparece.
    - Paso E: Indica si el texto borra un continuo (reduce un espectro a binario).
    - Paso F: Emite el dictamen final (“Sí”/“No”) y una justificación breve (2–4 frases) explicando por qué hay o no falso dilema.
    - Verificación final: respeta longitudes, español, estructura JSON exacta, sin texto fuera del JSON.
    
    CÁLCULO DE "confidence"
    - "confidence" es un valor continuo en [0.0, 1.0] que indica qué tan clara y segura es tu decisión sobre si hay “False Dilemma” o no.
    - Para estimarlo, combina dos factores:
        1. Claridad de la evidencia textual:
            Alto (≈0.8–1.0): la presencia/ausencia de “False Dilemma” es muy evidente, con patrones claros y consistentes, y casi sin ambigüedad.
            Medio (≈0.4–0.7): hay algunos indicios, pero también contraejemplos, ambigüedad o poca cantidad de evidencia.
            Bajo (≈0.0–0.3): el texto es confuso o casi no ofrece base para decidir.
        2. Seguridad interna del modelo:
            Si podrías haber elegido fácilmente la etiqueta contraria, baja el confidence. Si tu respuesta te parece la única opción razonable dadas las alternativas, sube el confidence.
    - Si el sistema subyacente dispone de probabilidades internas (por ejemplo, asociadas a distintas salidas posibles por beam search), usa también esas probabilidades como guía: distribuciones concentradas en una opción → confidence más alto; distribuciones repartidas entre opciones → confidence más bajo.

    ─────────────────────────────────────────────────────────────────────────────
    EJEMPLO GUIADO (one-shot). La salida del modelo DEBE SER SOLO el JSON.
    Entrada (TEXTO):
    "Referring to your claim that providing medicare for all citizens would be costly and a danger to the free market, I infer that you don’t care if people die from not having healthcare, so we are not going to support your endeavour."

    Traza de decisión resumida (NO imprimir en la salida):
    - A (opciones): se contraponen “apoyar medicare for all” vs. “no te importa si la gente muere”.
    - B (exclusividad): inferencia dicotómica que equipara oposición a una propuesta con indiferencia ante muertes; evidencia: "I infer that you don’t care if people die from not having healthcare".
    - C (terceras opciones): múltiples políticas intermedias (público/privado mixto, topes de precios, regulación antimonopolio, etc.).
    - D (cierre imperativo): "so we are not going to support your endeavour".
    - E (borra continuo): reduce un espectro de políticas sanitarias a dos polos.
    - F (dictamen): “Sí”; justificación breve; confianza alta por presencia de B, C, D y E.

    Salida esperada (solo JSON):
    {
      "is_fd": "Sí",
      "options_extracted": [
        "providing medicare for all citizens",
        "you don’t care if people die from not having healthcare"
      ],
      "exclusivity_claimed": {
        "value": true,
        "evidence": "I infer that you don’t care if people die from not having healthcare"
      },
      "third_options_possible": [
        "Reformas graduales de cobertura con subsidios y ampliación de programas existentes.",
        "Opción pública que compita con aseguradoras privadas.",
        "Topes a precios y control de costos hospitalarios y farmacéuticos.",
        "Regulación antimonopolio y estandarización de beneficios mínimos.",
        "Seguro catastrófico universal con redes de seguridad.",
        "Modelos mixtos (single-payer parcial, aseguramiento obligatorio, vouchers)."
      ],
      "imperative_closure": {
        "value": true,
        "evidence": "so we are not going to support your endeavour"
      },
      "erases_continuum": true,
      "justification": "Se fuerza una disyuntiva entre apoyar “medicare for all” o “no te importa si la gente muere”, eliminando alternativas intermedias plausibles. La conclusión operativa cierra opciones (“no vamos a apoyar…”). Esto configura un falso dilema que borra el continuo de políticas de salud.",
      "confidence": 0.9,
      "justificacion_confidence": "La evidencia es muy clara y la decisión “Sí” parece la única razonable. Esto sugiere un confidence alto."
    }
    """

    texto: str = dspy.InputField(desc="TEXTO a analizar, tal cual aparece.")
    salida: FalseDilemmaJudgment = dspy.OutputField(
        desc="Devuelve SOLO el objeto JSON válido conforme al esquema."
    )

class FalseDilemmaRunner(TechniqueRunner):
    """Detects the False Dilemma propaganda technique.

    Identifies arguments that present only two mutually exclusive options
    while ignoring intermediate or alternative possibilities, forcing a
    binary choice where none is warranted.

    Output dict keys (from postprocess):
        model: "FALSE_DILEMMA"
        answer: "Sí" or "No"
        rationale_summary: justification + options + exclusivity +
                           third_options + imperative_closure + erases_continuum
        confidence: float in [0.0, 1.0]
        span: exclusivity_claimed evidence + imperative_closure evidence
        labels: options_extracted, exclusivity_claimed, third_options_possible,
                imperative_closure, erases_continuum
        raw: full FalseDilemmaJudgment model dump
    """
    name = "FALSE_DILEMMA"
    signature = DetectaFalsoDilema

    def postprocess(self, salida_obj: FalseDilemmaJudgment) -> Dict[str, Any]:
        """Convert LLM judgment output to a standardized candidate dict.

        Args:
            salida_obj: Parsed Pydantic judgment object from the DSPy predictor.

        Returns:
            Dict with keys: model, answer (Sí/No), rationale_summary, confidence
            (float in [0, 1]), span (list of text fragments), labels (technique-
            specific metadata), and raw (full model dump).
        """
        answer = salida_obj.is_fd  # "Sí" / "No"

        # Spans literales principales (si existen)
        span: List[str] = []
        if salida_obj.exclusivity_claimed and salida_obj.exclusivity_claimed.evidence:
            span.append(salida_obj.exclusivity_claimed.evidence)
        if salida_obj.imperative_closure and salida_obj.imperative_closure.evidence:
            span.append(salida_obj.imperative_closure.evidence)
        # dedup + límite sobrio
        span = list(dict.fromkeys(span))[:3]

        parts: List[str] = []
        if salida_obj.justification:
            parts.append(salida_obj.justification.strip())

        if salida_obj.options_extracted:
            parts.append("Opciones planteadas: " + "; ".join(salida_obj.options_extracted[:3]) + ".")

        if salida_obj.exclusivity_claimed:
            parts.append(
                f"Exclusividad: {salida_obj.exclusivity_claimed.value}"
                + (f" (evidencia: {salida_obj.exclusivity_claimed.evidence})."
                   if salida_obj.exclusivity_claimed.evidence else ".")
            )

        if salida_obj.third_options_possible:
            parts.append(
                "Terceras opciones plausibles: "
                + "; ".join(salida_obj.third_options_possible[:4])
                + "."
            )

        if salida_obj.imperative_closure:
            parts.append(
                f"Cierre imperativo: {salida_obj.imperative_closure.value}"
                + (f" (evidencia: {salida_obj.imperative_closure.evidence})."
                   if salida_obj.imperative_closure.evidence else ".")
            )

        parts.append(f"Borra continuo: {bool(salida_obj.erases_continuum)}.")

        rationale_summary = " ".join(parts).strip()

        return {
            "model": self.name,
            "answer": answer,
            "rationale_summary": rationale_summary,
            "confidence": float(salida_obj.confidence),
            "span": span,

            "labels": {
                "options_extracted": salida_obj.options_extracted,
                "exclusivity_claimed": salida_obj.exclusivity_claimed.model_dump(),
                "third_options_possible": salida_obj.third_options_possible,
                "imperative_closure": salida_obj.imperative_closure.model_dump(),
                "erases_continuum": salida_obj.erases_continuum,
            },

            "raw": salida_obj.model_dump()
        }

    
# =========================================================
# 4) Registry de técnicas
# =========================================================

DEFAULT_TECHNIQUES: List[Type[TechniqueRunner]] = [
    RepetitionRunner,
    ExaggerationRunner,
    ObfuscationRunner,
    LoadedLanguageRunner,
    WhataboutismRunner,
    KairosRunner,
    ConversationKillerRunner,
    SlipperyRunner,
    SloganRunner,
    AppealToValuesRunner,
    RedHerringRunner,
    StrawmanRunner,
    FearPrejudiceRunner,
    AuthorityRunner,
    BandwagonRunner,
    CastingDoubtRunner,
    FlagWavingRunner,
    SmearPoisoningRunner,
    TuQuoqueRunner,
    GuiltByAssociationRunner,
    NameCallingRunner,
    CausalOversimplificationRunner,
    FalseDilemmaRunner
]


# =========================================================
# 5) Clase Analisis (orquesta los modelos base)
# =========================================================

class Analisis:
    """Orchestrates all propaganda technique detectors over a given text.

    Instantiates one runner per technique and executes them sequentially,
    collecting a list of candidate dicts. Failures in individual runners are
    caught and returned as error candidates rather than crashing the pipeline.

    Techniques included by default (DEFAULT_TECHNIQUES):
        Repetition, Exaggeration/Minimisation, Obfuscation, Loaded Language,
        Whataboutism, Kairos, Conversation Killer, Slippery Slope, Slogan,
        Appeal to Values, Red Herring, Strawman, Appeal to Fear/Prejudice,
        Appeal to Authority, Bandwagon, Casting Doubt, Flag Waving,
        Smear/Poisoning the Well, Tu Quoque, Guilt by Association,
        Name Calling, Causal Oversimplification, False Dilemma.

    Example:
        cfg = Configuracion(model_name="gpt-4o-2024-08-06", api_key=...)
        cfg.setup()
        analisis = Analisis()
        candidates = analisis.run(TEXTO)
    """

    def __init__(self, techniques: Optional[List[Type[TechniqueRunner]]] = None):
        """Initialize runners for each technique.

        Args:
            techniques: Optional list of TechniqueRunner subclasses to use.
                Defaults to DEFAULT_TECHNIQUES (all 23 detectors).
        """
        self.technique_classes = techniques or DEFAULT_TECHNIQUES
        self.runners = [cls() for cls in self.technique_classes]

    def run(self, texto: str) -> List[Dict[str, Any]]:
        """Run all technique detectors on the input text.

        Args:
            texto: The text to analyze for propaganda techniques.

        Returns:
            List of candidate dicts, one per technique. Each dict contains
            at minimum: model, answer, rationale_summary, confidence, span.
            Failed runners return a dict with error=True and confidence=0.0.
        """
        candidates = []
        for runner in self.runners:
            try:
                candidates.append(runner.run(texto))
            except Exception as e:
                # Fallback defensivo: si una técnica falla, no rompe el pipeline
                candidates.append({
                    "model": runner.name,
                    "answer": "No",
                    "rationale_summary": f"Fallo en ejecución de técnica {runner.name}: {str(e)[:180]}",
                    "confidence": 0.0,
                    "span": [],
                    "error": True
                })
        return candidates

# ====================================================================================================
# 5.5) Clase Consistency (corre N veces el análisis y entrega una respuesta consolidada y consistente)
# ====================================================================================================

class ConsistencyModule:
    """Runs the analysis pipeline multiple times and consolidates results by detection ratio.

    Executes ``analysis.run(TEXTO)`` for ``trials`` iterations and aggregates
    per-technique detections using a threshold on the detection ratio.

    Consolidation logic:
        - ratio = (number of trials where technique was detected) / trials
        - final answer = "Sí" if ratio >= threshold, else "No"
        - Among candidates matching the final answer, select the one with the
          highest confidence as the representative output.
        - Adds a ``ratio`` field to each consolidated candidate dict.

    Example:
        module = ConsistencyModule(analisis, trials=5, threshold=1.0)
        candidates = module.run(TEXTO)
    """

    def __init__(self, analysis_module, trials: int = 5, threshold: float = 1.0):
        """Initialize the consistency module.

        Args:
            analysis_module: An Analisis instance (or compatible object with a
                ``run(texto)`` method returning a list of candidate dicts).
            trials: Number of times to run the analysis. Minimum 1.
            threshold: Detection ratio threshold in [0, 1]. A technique is
                considered detected if its ratio >= threshold. Default 1.0
                (unanimous agreement across all trials).
        """
        self.analysis = analysis_module
        self.trials = max(1, int(trials))
        self.threshold = max(0, float(threshold))

    @staticmethod
    def _extract_candidates_from_output(out: Any) -> List[Dict[str, Any]]:
        """Extract a candidate list from the raw output of analysis.run().

        Args:
            out: Output from analysis.run() — either a list directly or a dict
                with a ``candidates`` key.

        Returns:
            List of candidate dicts, or empty list if extraction fails.
        """
        # Caso típico: dict con clave "candidates"
        if isinstance(out, dict):
            cands = out.get("candidates")
            if isinstance(cands, list):
                return cands
        # Alternativa: el propio output es la lista
        if isinstance(out, list):
            return out
        return []

    @staticmethod
    def _is_positive_answer(ans: Any) -> bool:
        """Interpret a candidate answer field as a positive (detected) boolean.

        Args:
            ans: Raw answer value — a bool or a string such as "Sí", "Yes", "True".

        Returns:
            True if the answer represents a positive detection, False otherwise.
        """
        # Heurística simple: interpreta Sí/Yes/True (bool o string)
        if isinstance(ans, bool):
            return ans
        s = str(ans).strip().lower()
        return s in {"si", "sí", "yes", "true", "y", "Sí", "SI", "YES", "Yes", "TRUE", "True", "Yes"}

    @staticmethod
    def _get_confidence(cand: Dict[str, Any]) -> float:
        """Extract the confidence score from a candidate dict.

        Args:
            cand: Candidate dict, optionally with a top-level ``confidence``
                or a nested ``raw.confidence`` fallback.

        Returns:
            Confidence as a float. Returns 0.0 if absent or non-numeric.
        """
        conf = cand.get("confidence", None)
        if conf is None:
            raw = cand.get("raw") or {}
            conf = raw.get("confidence", 0.0)
        try:
            return float(conf)
        except Exception:
            return 0.0

    def run(self, TEXTO: str) -> List[Dict[str, Any]]:
        """Run the analysis multiple times and return consolidated candidates.

        Args:
            TEXTO: The text to analyze for propaganda techniques.

        Returns:
            List of consolidated candidate dicts, one per technique. Each dict
            contains the same fields as a standard candidate output plus a
            ``ratio`` field (float in [0, 1]) indicating the detection rate
            across trials.
        """
        trials = self.trials
        threshold = self.threshold

        # Estructura: por técnica/model -> info agregada
        agg: Dict[str, Dict[str, Any]] = {}

        for t in range(trials):
            out = self.analysis.run(TEXTO)
            candidates = self._extract_candidates_from_output(out)

            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                model = cand.get("model")
                if not model:
                    continue
                key = str(model)

                if key not in agg:
                    agg[key] = {
                        "cands": [],
                        "per_trial_positive": [False] * trials,
                        "per_trial_any": [False] * trials,
                    }

                agg[key]["cands"].append(cand)

                is_pos = self._is_positive_answer(cand.get("answer"))
                agg[key]["per_trial_any"][t] = True
                # Si hay más de un candidato por técnica en un mismo trial, hacemos OR
                agg[key]["per_trial_positive"][t] = (
                    agg[key]["per_trial_positive"][t] or is_pos
                )

        consolidated: List[Dict[str, Any]] = []

        for model, info in agg.items():
            cands = info["cands"]
            per_pos = info["per_trial_positive"]
            detected_count = sum(1 for v in per_pos if v)
            ratio = detected_count / float(trials)

            # Regla de decisión global
            target_answer_positive = ratio >= float(threshold)

            # Elegimos el mejor candidate según answer objetivo
            pool = [
                c for c in cands
                if self._is_positive_answer(c.get("answer")) == target_answer_positive
            ]
            if not pool:
                # Fallback: si por alguna razón no hay candidates con esa polaridad,
                # usamos todos los disponibles.
                pool = cands

            best_cand = max(pool, key=self._get_confidence)
            best_conf = self._get_confidence(best_cand)

            merged = deepcopy(best_cand)
            merged["answer"] = "Sí" if target_answer_positive else "No"
            merged["confidence"] = best_conf
            merged["ratio"] = ratio

            raw = merged.get("raw") or {}
            if not isinstance(raw, dict):
                raw = {"_raw_original": raw}
            # Guardamos también el ratio dentro de raw, por si quieres depurar
            raw.setdefault("consistency_ratio", ratio)
            raw.setdefault("confidence", best_conf)
            merged["raw"] = raw

            consolidated.append(merged)

        return consolidated


def run_consistency(analysis_module, TEXTO: str, trials: int = 5, threshold: float = 1.0) -> List[Dict[str, Any]]:
    """Convenience wrapper for ConsistencyModule.

    Runs the analysis pipeline ``trials`` times and consolidates per-technique
    detections by majority-vote ratio threshold.

    Args:
        analysis_module: An Analisis instance with a ``run(texto)`` method.
        TEXTO: The text to analyze for propaganda techniques.
        trials: Number of independent analysis runs. Default 5.
        threshold: Minimum detection ratio to classify a technique as detected.
            Default 1.0 (unanimous across all trials).

    Returns:
        List of consolidated candidate dicts (one per technique), each
        including a ``ratio`` field indicating the detection rate.
    """
    return ConsistencyModule(analysis_module, trials=trials, threshold=threshold).run(TEXTO)


# =========================================================
# 6) Juez
# =========================================================

# =========================
# Pydantic schemas (guardrails)
# =========================

class CandidateScore(BaseModel):
    """Rubric scores and inclusion decision for a single candidate technique."""
    scores: Dict[str, int] = Field(default_factory=dict)   # correccion, consistencia, etc.
    total: float = 0.0
    include: bool = False
    strengths: List[str] = Field(default_factory=list)
    dependency_trace: Optional[List[str]] = None


class JudgeScoresOutput(BaseModel):
    """Container for the judge's rubric scores across all candidate techniques."""
    candidates: List[CandidateScore] = Field(default_factory=list)


class JudgeOutputNormalized(BaseModel):
    """Normalized output from the JudgeModule after scoring and selection."""
    selected_indices: List[int] = Field(default_factory=list)
    scores: JudgeScoresOutput = Field(default_factory=JudgeScoresOutput)
    brief_justification: str = ""


class SynthesisOutputNormalized(BaseModel):
    """Normalized output from the SynthesizeModule with the final consolidated answer."""
    final_answer: str = ""
    brief_reasoning: str = ""


class SelectionOutputNormalized(BaseModel):
    """Post-hoc selection of detected technique models, with per-model evidence and confidence."""
    selected_models: List[str] = Field(default_factory=list)
    evidence: Dict[str, str] = Field(default_factory=dict)
    confidence: confloat(ge=0.0, le=1.0) = 0.0


# =========================
# DSPy Signatures
# =========================

class JudgeSignature(dspy.Signature):
    """Evalúa y RANKEA candidatos; marca cuáles deben incluirse."""
    question: str = dspy.InputField(desc="pregunta original")
    candidates_json: str = dspy.InputField(
        desc="lista JSON con objetos {model, answer, rationale_summary, confidence, ratio}"
    )
    rubric: str = dspy.InputField(desc="criterios, ponderaciones y REGLAS DE INCLUSIÓN")

    selected_indices_json: str = dspy.OutputField(desc="JSON lista de índices seleccionados [int]")
    scores_json: str = dspy.OutputField(desc="JSON con puntajes por candidato y criterio (+ flags include)")
    brief_justification: str = dspy.OutputField(
        desc="justificación concisa (≤3 frases), menciona MODELOS en MAYÚSCULAS y principios usados"
    )


class SynthesizeSignature(dspy.Signature):
    """Produce respuesta final unificada SIN descartar 'must-include'."""
    question: str = dspy.InputField()
    topk_json: str = dspy.InputField(desc="JSON con candidatos seleccionados (incluye flags include y strengths)")
    final_answer: str = dspy.OutputField(desc="respuesta clara (≤6 frases) listando TODAS las técnicas válidas")
    brief_reasoning: str = dspy.OutputField(desc="justificación breve (≤4 frases), menciona MODELOS en MAYÚSCULAS")


class SelectJudgeCandidates(dspy.Signature):
    """
    Lee final_answer (ES) y candidates (JSON list). Devuelve modelos seleccionados por el juez.
    """
    final_answer: str = dspy.InputField(desc="Resumen en lenguaje natural de las técnicas detectadas por el juez.")
    candidates_json: str = dspy.InputField(desc="JSON (lista) de candidates con campos 'model', 'answer', etc.")
    output: str = dspy.OutputField(desc="JSON con selected_models, evidence y confidence.")


# =========================
# Helpers
# =========================

def safe_json_loads(x: Any, default):
    """Parse a JSON string safely, returning a default value on failure.

    Args:
        x: Value to parse. Returned as-is if already a dict or list.
        default: Value to return if parsing fails or input is None.

    Returns:
        Parsed object, the original value (if already dict/list), or ``default``.
    """
    if x is None:
        return default
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default


def normalize_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normaliza spans y rationale para evitar crashes."""
    normed = []
    for c in candidates:
        c2 = dict(c)

        if c2.get("rationale_summary") is None:
            c2["rationale_summary"] = ""

        sp = c2.get("span")
        if sp is None:
            c2["span"] = []
        elif isinstance(sp, str):
            c2["span"] = [sp]
        elif isinstance(sp, list):
            c2["span"] = sp
        else:
            c2["span"] = []

        # Asegura campos mínimos esperados por el juez
        if "model" not in c2:
            c2["model"] = "UNKNOWN"
        if "answer" not in c2:
            c2["answer"] = "No"
        if "confidence" not in c2:
            c2["confidence"] = 0.0
        if "ratio" not in c2:
            c2["ratio"] = 0.0

        normed.append(c2)
    return normed


def normalize_judge_output(j_pred, num_candidates: int) -> JudgeOutputNormalized:
    """
    Convierte salidas string del LLM a estructura consistente.

    IMPORTANTE:
    - Los índices seleccionados se derivan EXCLUSIVAMENTE de los flags `include`
      retornados en `scores.candidates`, que son el resultado de aplicar DEFAULT_RUBRIC.
    - No se utiliza `selected_indices_json` para decidir qué técnicas quedan seleccionadas.
    """
    scores_raw = safe_json_loads(getattr(j_pred, "scores_json", None), default={})
    try:
        scores_norm = JudgeScoresOutput.model_validate(scores_raw)
    except ValidationError:
        # Si no hay estructura válida, no seleccionamos nada (la rúbrica falla)
        scores_norm = JudgeScoresOutput(
            candidates=[CandidateScore() for _ in range(num_candidates)]
        )

    # Selección SOLO por rúbrica: índices donde include==True
    selected: List[int] = []
    for idx, cand_sc in enumerate(scores_norm.candidates):
        if isinstance(cand_sc, CandidateScore) and cand_sc.include:
            if 0 <= idx < num_candidates:
                selected.append(idx)

    # No aplicamos ningún fallback heurístico: si la rúbrica no selecciona nada,
    # `selected_indices` queda vacío, y el pipeline reflejará "sin técnicas seleccionadas".
    return JudgeOutputNormalized(
        selected_indices=selected,
        scores=scores_norm,
        brief_justification=(getattr(j_pred, "brief_justification", "") or "").strip(),
    )


def normalize_synth_output(s_pred) -> SynthesisOutputNormalized:
    """Convert raw SynthesizeSignature LLM output to a normalized structure.

    Args:
        s_pred: DSPy prediction object from SynthesizeSignature with
            ``final_answer`` and ``brief_reasoning`` attributes.

    Returns:
        SynthesisOutputNormalized with stripped string fields.
    """
    return SynthesisOutputNormalized(
        final_answer=(getattr(s_pred, "final_answer", "") or "").strip(),
        brief_reasoning=(getattr(s_pred, "brief_reasoning", "") or "").strip(),
    )


def normalize_selection(raw_output: str, candidates: List[Dict[str, Any]]) -> SelectionOutputNormalized:
    """Parse and validate the post-hoc model selection output from the LLM.

    Robustly handles string, list, or dict formats for ``selected_models``
    and ``evidence``. Filters results to only include models present in
    ``candidates``.

    Args:
        raw_output: JSON string output from SelectJudgeCandidates signature.
        candidates: Full list of candidate dicts used to validate model names.

    Returns:
        SelectionOutputNormalized with validated selected_models, per-model
        evidence strings, and an overall confidence score.
    """
    data = safe_json_loads(raw_output, default={})
    candidate_models = {c.get("model") for c in candidates if isinstance(c, dict)}

    # ---- selected_models robusto ----
    selected_raw = data.get("selected_models", [])
    if isinstance(selected_raw, str):
        selected_raw = safe_json_loads(selected_raw, default=[])
        if isinstance(selected_raw, str):
            selected_raw = [s.strip() for s in selected_raw.split(",") if s.strip()]
    if not isinstance(selected_raw, list):
        selected_raw = []
    selected = [m for m in selected_raw if m in candidate_models]

    # ---- evidence robusto ----
    evidence_raw = data.get("evidence") or {}
    if isinstance(evidence_raw, str):
        evidence_raw = safe_json_loads(evidence_raw, default={})
    if isinstance(evidence_raw, list):
        evidence_raw = {
            d.get("model"): d.get("evidence")
            for d in evidence_raw
            if isinstance(d, dict) and d.get("model") in candidate_models
        }
    if not isinstance(evidence_raw, dict):
        evidence_raw = {}

    evidence = {
        k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        for k, v in evidence_raw.items()
        if k in candidate_models
    }

    conf_raw = data.get("confidence", 0.0)
    try:
        conf = float(conf_raw)
    except Exception:
        conf = 0.0

    try:
        return SelectionOutputNormalized(
            selected_models=selected,
            evidence=evidence,
            confidence=conf,
        )
    except ValidationError:
        return SelectionOutputNormalized()


# =========================
# DSPy Modules
# =========================

class JudgeModule(dspy.Module):
    """DSPy module that scores and ranks candidate technique detections using a rubric.

    Wraps JudgeSignature to evaluate all candidate outputs and determine which
    techniques should be included in the final answer based on rubric criteria.
    """

    def __init__(self):
        """Initialize the judge predictor."""
        super().__init__()
        self.judge = dspy.Predict(JudgeSignature)

    def forward(self, question: str, candidates: List[Dict[str, Any]], rubric: str) -> JudgeOutputNormalized:
        """Score candidates against the rubric and select which to include.

        Args:
            question: The analysis question fed to the LLM judge.
            candidates: List of candidate dicts from TechniqueRunner outputs.
            rubric: Scoring rubric string (DEFAULT_RUBRIC is always used).

        Returns:
            JudgeOutputNormalized with selected_indices, scores, and brief_justification.
        """
        candidates = normalize_candidates(candidates)
        j = self.judge(
            question=question,
            candidates_json=json.dumps(candidates, ensure_ascii=False),
            rubric=rubric
        )
        return normalize_judge_output(j, num_candidates=len(candidates))


class SynthesizeModule(dspy.Module):
    """DSPy module that produces a unified natural-language synthesis from top-k candidates.

    Takes the judge-selected candidates and generates a final consolidated answer
    listing all detected propaganda techniques with a brief reasoning.
    """

    def __init__(self):
        """Initialize the synthesis predictor."""
        super().__init__()
        self.synth = dspy.Predict(SynthesizeSignature)

    def forward(self, question: str, topk: List[Dict[str, Any]]) -> SynthesisOutputNormalized:
        """Generate the final consolidated answer from top-k selected candidates.

        Args:
            question: The analysis question.
            topk: Judge-selected candidate dicts (include=True).

        Returns:
            SynthesisOutputNormalized with final_answer and brief_reasoning.
        """
        s = self.synth(
            question=question,
            topk_json=json.dumps(topk, ensure_ascii=False)
        )
        return normalize_synth_output(s)


class SelectJudgeCandidatesModule(dspy.Module):
    """DSPy module that maps the synthesized answer back to specific candidate models.

    Performs a post-hoc selection step: reads the final_answer text and the
    candidates list to identify which technique models were actually detected,
    returning per-model evidence and an overall confidence score.
    """

    def __init__(self):
        """Initialize the selector predictor."""
        super().__init__()
        self.selector = dspy.Predict(SelectJudgeCandidates)

    def forward(self, final_answer: str, candidates: List[Dict[str, Any]]) -> SelectionOutputNormalized:
        """Select candidate models mentioned in the final answer.

        Args:
            final_answer: Natural-language synthesis from SynthesizeModule.
            candidates: Full list of candidate dicts from Analisis.run().

        Returns:
            SelectionOutputNormalized with selected_models, evidence dict,
            and confidence. Results are filtered to only include models
            permitted by the judge rubric.
        """
        pred = self.selector(
            final_answer=final_answer,
            candidates_json=json.dumps(candidates, ensure_ascii=False)
        )
        return normalize_selection(pred.output, candidates)


JUDGE_MOD: Optional[JudgeModule] = None
SYNTH_MOD: Optional[SynthesizeModule] = None
SELECTOR_MOD: Optional[SelectJudgeCandidatesModule] = None


def get_modules():
    """Lazy init de módulos DSPy para no reinstanciarlos en cada llamada."""
    global JUDGE_MOD, SYNTH_MOD, SELECTOR_MOD
    if JUDGE_MOD is None:
        JUDGE_MOD = JudgeModule()
    if SYNTH_MOD is None:
        SYNTH_MOD = SynthesizeModule()
    if SELECTOR_MOD is None:
        SELECTOR_MOD = SelectJudgeCandidatesModule()
    return JUDGE_MOD, SYNTH_MOD, SELECTOR_MOD


def RunJuez(TEXTO: str, candidates: List[Dict[str, Any]], rubric: Optional[str] = None) -> Dict[str, Any]:
    """Run the full judge pipeline: scoring, synthesis, and post-hoc selection.

    Executes three sequential steps:
        1. JudgeModule: scores candidates against DEFAULT_RUBRIC and selects
           which techniques to include (``include=True`` flag).
        2. SynthesizeModule: generates a natural-language answer from top-k
           selected candidates.
        3. SelectJudgeCandidatesModule: maps the synthesis back to specific
           model names, filtered to only rubric-approved techniques.

    Note:
        The ``rubric`` parameter is accepted for API compatibility but ignored;
        DEFAULT_RUBRIC is always used to ensure consistent evaluation.

    Args:
        TEXTO: The original text that was analyzed.
        candidates: List of candidate dicts from Analisis.run() or
            run_consistency().
        rubric: Ignored. DEFAULT_RUBRIC is always applied.

    Returns:
        Dict with keys:
            judged: JudgeOutputNormalized model dump (scores, selected_indices,
                brief_justification).
            topk: List of selected candidate dicts with rubric score info merged in.
            synthesis: SynthesisOutputNormalized model dump (final_answer,
                brief_reasoning).
            selected_models_posthoc: SelectionOutputNormalized model dump
                (selected_models, evidence, confidence).
            elapsed_sec: Wall-clock time for the full judge pipeline.
    """
    start = time.time()

    candidates = normalize_candidates(candidates)
    judge_mod, synth_mod, selector_mod = get_modules()

    question = (
        "23 modelos analizarán cada técnica y te entregarán su respuesta junto con confidence y ratio. "
        "Tu tarea es analizar las 23 respuestas y entregar un resultado consolidado. "
        f"TEXTO: {TEXTO}"
    )

    # Siempre usamos DEFAULT_RUBRIC, ignorando el parámetro rubric externo.
    judged = judge_mod.forward(
        question=question,
        candidates=candidates,
        rubric=DEFAULT_RUBRIC
    )

    scores_list = judged.scores.candidates or []

    # topk: solo índices con include=True según la rúbrica
    topk: List[Dict[str, Any]] = []
    for i in judged.selected_indices:
        base = candidates[i]
        extra = scores_list[i].model_dump() if i < len(scores_list) else {}
        topk.append({**base, **extra})

    synth_out = synth_mod.forward(
        question=question,
        topk=topk
    )

    # Selección post-hoc basada en razonamiento,
    # pero restringida EXCLUSIVAMENTE a modelos seleccionados por la rúbrica.
    selection_out = selector_mod.forward(
        final_answer=synth_out.brief_reasoning or synth_out.final_answer,
        candidates=candidates
    )

    # Conjunto de modelos permitidos por rúbrica
    allowed_models = {
        candidates[i].get("model")
        for i in judged.selected_indices
        if 0 <= i < len(candidates)
    }

    # Filtra selected_models y evidence para que SOLO queden técnicas permitidas por la rúbrica
    selection_out.selected_models = [
        m for m in selection_out.selected_models
        if m in allowed_models
    ]
    selection_out.evidence = {
        k: v for k, v in selection_out.evidence.items()
        if k in allowed_models
    }

    end = time.time()

    return {
        "judged": judged.model_dump(),
        "topk": topk,
        "synthesis": synth_out.model_dump(),
        "selected_models_posthoc": selection_out.model_dump(),
        "elapsed_sec": end - start,
    }


# ---------------------------------------------------------------------
# Rúbrica por defecto del juez (módulo)
# ---------------------------------------------------------------------

DEFAULT_RUBRIC = r'''Evalúa con esta rúbrica y devuelve JSON estructurado.

Criterios y ponderaciones (suman 1.0):
- Consistencia (0.30)
- Seguridad / No alucinar (0.35)
- PreferenciaSí (0.35)

Mapeo "Consistencia" (según ratio):
- 5 si ratio ≥ 0.95
- 4 si 0.90 ≤ ratio < 0.95
- 3 si 0.80 ≤ ratio < 0.90
- 2 si 0.70 ≤ ratio < 0.80
- 1 si ratio < 0.70

Mapeo "Seguridad / No alucinar" (según confidence):
- 5 si confidence ≥ 0.95
- 4 si 0.90 ≤ confidence < 0.95
- 3 si 0.80 ≤ confidence < 0.90
- 2 si 0.70 ≤ confidence < 0.80
- 1 si confidence < 0.70

Mapeo "PreferenciaSí":
- 5 si answer afirma la técnica (Sí/True/etc.)
- 3 si es condicional/ambiguo (Parcial/Depende) con justificación breve.
- 1 si niega (No/False/etc.)

Cálculo:
1) Para cada candidato, asigna notas 1–5 en cada criterio.
2) Normaliza dividiendo por 5 y calcula "total" = Σ (nota_norm * peso).
3) Marca include=true si y solo si total > 0.95

Empates y ranking:
- Ordena por total, luego Seguridad, luego PreferenciaSí, luego Consistencia.
- Devuelve selected_indices_json con TODOS los índices donde total > 0.95
- En scores_json, por candidato incluye:
  - "scores": {"consistencia": 1-5, "seguridad": 1-5, "preferencia_si": 1-5}, "total": float, "include": true/false,
  - "strengths": ["False Dilemma: '...'","Flag Waving: '...'"] si las extrae del rationale/answer.

Las técnicas tienen ciertas dependencias entre ellas. Considéralas para abordar ambigüedades o dudas que surjan durante el análisis. 

    - STRAWMAN es un tipo de RED HERRING enfocado en el argumento del oponente. 
    - WHATABOUTISM es un tipo de STRAWMAN (réplica con contra-pregunta) y también es un tipo de CASTING DOUBT (desvía del tópico original)
    - APPEAL TO HYPOCRISY es un tipo de CASTING DOUBT (apunta a contradicción pasado/presente del sujeto)
    - QUESTIONING THE REPUTATION es un tipo de CASTING DOUBT (moral qualities del sujeto)
    - GUILT BY ASSOCIATION es un tipo de NAME CALLING (aplicado a una idea)
    - NAME CALLING es un tipo de LOADED LANGUAGE (centrado en el sujeto)
    - CONVERSATION KILLER es un tipo de LOADED LANGUAGE (para terminar el intercambio)
    - APPEAL TO FEAR es un tipo de LOADED LANGUAGE (descripción atemorizante)
    - LOADED LANGUAGE es un tipo de APPEAL TO VALUES (invoca valores sin anclaje contextual)
    - FLAG WAVING es un tipo de APPEAL TO VALUES (exalta orgullo/cohesión grupal)

    Co-ocurrencias (boosts inferenciales, no obligación)
    - APPEAL TO FEAR tiende a co-ocurrir con CONSEQUENTIAL OVERSIMPLIFICATION (miedo apoyado en consecuencias simplificadas)
    - APPEAL TO FEAR tiende a co-ocurrir con FALSE DILEMMA (dos opciones exclusivas para intensificar el miedo)

    Principios de consolidación
    1) Herencia ascendente (implicación): si hay un subtipo confirmado (p.ej., STRAWMAN como tipo de RED HERRING), su(s) padre(s) están implicados pero no deben reemplazar al subtipo. Reporta el subtipo como principal; los padres se listan en dependency_trace.
    2) No-duplicación padre/hijo: evita listar simultáneamente padre y subtipo como hallazgos separados salvo que el texto contenga evidencia independiente para el padre más allá de la que sustenta al hijo.
    3) Múltiples padres posibles (herencia múltiple): WHATABOUTISM puede heredar de STRAWMAN y CASTING DOUBT. Si la evidencia apoya ambos, conserva WHATABOUTISM como técnica principal y anota a ambos padres en dependency_trace.
    4) Co-ocurrencia informada: si hay evidencia de APPEAL TO FEAR y aparecen patrones de consecuencias binarias o sobre-simplificadas, incrementa la prioridad de FALSE DILEMMA o CONSEQUENTIAL OVERSIMPLIFICATION si hay señales textuales mínimas (p.ej., “only two choices”, “inevitable outcome”). No infieras sin evidencia.
    5) Resolución de colisiones: si dos técnicas explican la misma evidencia, prefiere la más específica (subtipo) y degrada la más general a “apoyada por herencia”. Reporta el subtipo como principal; los padres se listan en dependency_trace.
    6) Criterio de contexto de valores: LOADED LANGUAGE requiere lenguaje valorativo/emotivo desanclado del contenido factual; si los valores están contextualizados con evidencia sustantiva, reduce su peso.
    

Formato de salida:
{
  "candidates": [
    {
      "model": "NOMBRE_MODELO",
      "answer": "...",
      "scores": {"consistencia": 1-5, "seguridad": 1-5, "preferencia_si": 1-5},
      "total": 0.x,
      "include": true/false,
      "strengths": ["False Dilemma: '...'","Flag Waving: '...'"]
    }, ...
  ]
}

brief_justification: 2–4 frases, menciona MODELOS en MAYÚSCULAS y explica por qué TODOS los seleccionados pasan el umbral/criterios.'''


# =========================================================
# 7) Visualizar los spans
# =========================================================

MODEL_COLORS: Dict[str, str] = {
    "EXAGERATION": "#e76f51", "LOADED_LANGUAGE": "#2a9d8f", "STRAWMAN": "#f4a261",
    "RED_HERRING": "#e9c46a", "REPETITION": "#264653", "KAIROS": "#8ab17d",
    "KILLER": "#b5838d", "SLIPPERY": "#7b9acc", "SLOGAN": "#b5e48c",
    "VALUES": "#48cae4", "FEAR": "#ef476f", "AUTHORITY": "#118ab2",
    "BANDWAGON": "#06d6a0", "CASTING_DOUBT": "#8338ec", "FLAG_WAVING": "#ff006e",
    "SMEAR": "#6d6875", "HYPOCRISY": "#adb5bd", "ASSOCIATION": "#bc6c25",
    "NAME_CALLING": "#457b9d", "OVERSIMPLIFICATION": "#e5989b", "FALSE_DILEMMA": "#90be6d",
    "OBFUSCATION": "#4d908e", "WHATABOUTISM": "#577590",
}
    
TRANS = str.maketrans({
    "’": "'", "‘": "'", "´": "'", "“": '"', "”": '"',
    "—": "-", "–": "-", "\u00A0": " ", "\u2009": " ", "\u200A": " ", "\u202F": " ",
})
TRAIL_PUNCT = ".,;:!?)»”]›>"

def norm_label(lbl: str) -> str:
    """Normaliza etiqueta de técnica a formato 'STRAWMAN', 'FALSE_DILEMMA', etc."""
    return re.sub(r"[\s\-]+", "_", (lbl or "").strip()).upper()

def strip_trailing_punct(s: str) -> str:
    """Remove trailing punctuation characters from a string.

    Args:
        s: Input string to strip.

    Returns:
        String with trailing punctuation (.,;:!?)»"›>) removed.
    """
    s = (s or "").rstrip()
    while s and s[-1] in TRAIL_PUNCT:
        s = s[:-1]
    return s

def norm_with_spans(s: str) -> Tuple[str, List[Tuple[int, int]], str]:
    """
    Normaliza un texto y devuelve:
    - texto normalizado,
    - mapeo de spans normalizados->originales (por carácter),
    - texto original ya NFKC+TRANS.
    """
    s0 = unicodedata.normalize("NFKC", s or "").translate(TRANS)
    out, spans = [], []
    i, n = 0, len(s0)
    while i < n:
        ch = s0[i]
        if ch.isspace():
            j = i + 1
            while j < n and s0[j].isspace():
                j += 1
            out.append(" ")
            spans.append((i, j))
            i = j
        else:
            out.append(ch)
            spans.append((i, i + 1))
            i += 1
    return "".join(out), spans, s0

def find_occurrences_robust(text: str, phrase: str) -> List[Tuple[int, int]]:
    """
    Busca ocurrencias robustas:
    1) búsqueda directa case-insensitive,
    2) si falla: normaliza frase y texto, quita puntuación final,
       y mapea a offsets originales.
    """
    if not phrase:
        return []
    # 1) directo
    patt = re.escape(phrase)
    hits = [(m.start(), m.end()) for m in re.finditer(patt, text, flags=re.IGNORECASE)]
    if hits:
        return hits

    # 2) normalizado
    phrase2 = strip_trailing_punct(unicodedata.normalize("NFKC", phrase).translate(TRANS))
    nt, spans, _ = norm_with_spans(text)
    np_, _, _ = norm_with_spans(phrase2)
    if not np_.strip():
        return []

    patt2 = re.escape(np_)
    hits2 = [(m.start(), m.end()) for m in re.finditer(patt2, nt, flags=re.IGNORECASE)]
    mapped = []
    for a, b in hits2:
        orig_start = spans[a][0]
        orig_end = spans[b - 1][1] if (b - 1) < len(spans) else len(text)
        mapped.append((orig_start, orig_end))
    return mapped

def normalize_spans_field(span_field: Any) -> List[str]:
    """span puede ser None, str, list[str]."""
    if span_field is None:
        return []
    if isinstance(span_field, str):
        return [span_field] if span_field.strip() else []
    if isinstance(span_field, list):
        return [s for s in span_field if isinstance(s, str) and s.strip()]
    return []

# ---- Entidades solo con técnicas seleccionadas ----
def collect_ents_selected_only(text: str,
                               candidates: List[Dict[str, Any]],
                               selection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find span occurrences in text for judge-selected techniques only.

    For each selected model, locates all span strings from the candidate dict
    in the original text using robust fuzzy matching.

    Args:
        text: The original text that was analyzed.
        candidates: Full list of candidate dicts from Analisis.run().
        selection: SelectionOutputNormalized model dump with ``selected_models``.

    Returns:
        Sorted list of entity dicts with keys ``start``, ``end``, ``label``
        (character offsets into ``text``).
    """
    selected = {norm_label(x) for x in (selection.get("selected_models") or [])}
    ents: List[Dict[str, Any]] = []
    for c in candidates:
        model = norm_label(c.get("model", ""))
        if model not in selected:
            continue
        for span_text in normalize_spans_field(c.get("span")):
            for s, e in find_occurrences_robust(text, span_text):
                ents.append({"start": s, "end": e, "label": model})
    ents.sort(key=lambda x: (x["start"], x["end"]))
    return ents

def segmentize_by_boundaries(text: str,
                             ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split text into contiguous segments based on entity start/end boundaries.

    Creates non-overlapping segments that cover the full text, where each
    segment tracks which technique labels are active within it.

    Args:
        text: The original text.
        ents: List of entity dicts with ``start``, ``end``, ``label`` keys.

    Returns:
        List of segment dicts with keys ``start``, ``end``, ``labels``
        (list of active technique label strings).
    """
    boundaries = {0, len(text)}
    for e in ents:
        boundaries.add(e["start"])
        boundaries.add(e["end"])
    pts = sorted(boundaries)
    segments: List[Dict[str, Any]] = []
    for i in range(len(pts) - 1):
        s, e = pts[i], pts[i + 1]
        if s >= e:
            continue
        active = {
            en["label"] for en in ents
            if not (en["end"] <= s or en["start"] >= e)
        }
        segments.append({"start": s, "end": e, "labels": sorted(active)})
    return segments

def mask_segments_to_selected(segments: List[Dict[str, Any]],
                              selection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter segment labels to only include judge-selected technique models.

    Args:
        segments: List of segment dicts from ``segmentize_by_boundaries``.
        selection: SelectionOutputNormalized model dump with ``selected_models``.

    Returns:
        New list of segment dicts with labels filtered to permitted models only.
    """
    allowed = {norm_label(x) for x in (selection.get("selected_models") or [])}
    masked = []
    for seg in segments:
        labs = [l for l in seg["labels"] if l in allowed]
        masked.append({"start": seg["start"], "end": seg["end"], "labels": labs})
    return masked

# ---- Render HTML ----
def gradient_for_labels(labels: List[str]) -> str:
    """Build a CSS color or linear-gradient string for a list of technique labels.

    Args:
        labels: List of normalized technique label strings (e.g. ``["STRAWMAN"]``).

    Returns:
        A CSS color string for single labels, a ``linear-gradient(...)`` for
        multiple labels, or ``"transparent"`` if the list is empty.
    """
    if not labels:
        return "transparent"
    if len(labels) == 1:
        return MODEL_COLORS.get(labels[0], "#999")
    k, stops = len(labels), []
    for i, lab in enumerate(labels):
        c = MODEL_COLORS.get(lab, "#999")
        a, b = int(100 * i / k), int(100 * (i + 1) / k)
        stops.append(f"{c} {a}%, {c} {b}%")
    return f"linear-gradient(to bottom, {', '.join(stops)})"

def render_overlaps_inline_html(
    text: str,
    segments: List[Dict[str, Any]],
    show_legend: str = "bottom",   # "bottom"|"top"|""|None
    show_badges: bool = False,
    badge_style: str = "outline",  # "outline"|"solid"
    boxed: bool = True,
    title: str = "Spans resaltados",
    subtitle: Optional[str] = None,
    icon: str = "⚖️",
    accent_color: str = "#111827"
) -> str:
    """Devuelve un HTML string que resalta spans (NO hace display)."""

    def badge_html(label: str, color: str) -> str:
        if badge_style == "solid":
            return (f'<span style="background:{color}; color:#fff; padding:0 6px; '
                    f'border-radius:3px; font-size:11px; margin-right:6px; display:inline-block;">{label}</span>')
        return (f'<span style="background:transparent; color:{color}; padding:0 6px; '
                f'border:1px solid {color}; border-radius:3px; font-size:11px; '
                f'margin-right:6px; display:inline-block;">{label}</span>')

    parts: List[str] = []
    for seg in segments:
        s, e, labels = seg["start"], seg["end"], seg["labels"]
        frag = (text[s:e]
                .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br>"))

        if not labels:
            parts.append(frag)
            continue

        bg = gradient_for_labels(labels)
        title_tip = ", ".join(labels)
        text_style = (
            f"background:{bg}; padding:0 3px; border-radius:3px; "
            f"border:1px solid rgba(0,0,0,0.08); display:inline-block;"
        )

        if show_badges:
            row = " ".join(badge_html(l, MODEL_COLORS.get(l, "#999")) for l in labels)
            wrapped = (
                f'<span title="{title_tip}" style="display:inline-grid; grid-auto-flow:row; '
                f'justify-items:start; gap:2px; line-height:1.2; margin:0 2px;">'
                f'  <span style="{text_style}">{frag}</span>'
                f'  <span style="line-height:1; white-space:nowrap;">{row}</span>'
                f'</span>'
            )
        else:
            wrapped = f'<span title="{title_tip}" style="{text_style} margin:0 2px;">{frag}</span>'
        parts.append(wrapped)

    legend_html = ""
    if show_legend:
        labels_used = set()
        for seg in segments:
            labels_used.update(seg["labels"])
        if labels_used:
            chips = " ".join(
                f'<span style="background:{MODEL_COLORS.get(l, "#999")}; color:#fff; '
                f'padding:1px 6px; border-radius:3px; margin:0 6px 6px 0; '
                f'display:inline-block; font-size:12px;">{l}</span>'
                for l in sorted(labels_used)
            )
            legend_html = f'<div style="margin-top:10px; font-size:13px;"><b>Techniques:</b><br>{chips}</div>'
            if show_legend == "top":
                parts.insert(0, legend_html + "<br>")
                legend_html = ""

    base_style = "font-family:ui-sans-serif,system-ui; line-height:1.55; font-size:15px;"
    if boxed:
        container_style = (
            f"border:2px solid {accent_color}; border-radius:10px; background:#fafafa;"
            f"padding:0; margin:14px 0; box-shadow:0 2px 6px rgba(0,0,0,0.06); {base_style}"
        )
        header_style = (
            f"background:{accent_color}; color:white; padding:10px 14px; "
            f"border-top-left-radius:10px; border-top-right-radius:10px;"
            f"display:flex; align-items:center; gap:10px; font-weight:600;"
        )
        title_block = f"""
        <div style="{header_style}">
          <div style="font-size:18px; line-height:1;">{icon}</div>
          <div style="display:flex; flex-direction:column;">
            <div style="font-size:15px;">{title}</div>
            {f'<div style="font-size:12px; font-weight:400; opacity:.95;">{subtitle}</div>' if subtitle else ''}
          </div>
        </div>
        """
        body_style = "padding:14px 16px;"
        html = (
            f'<div style="{container_style}">'
            f'{title_block}'
            f'<div style="{body_style}">' + "".join(parts) + legend_html + "</div>"
            f'</div>'
        )
    else:
        html = f'<div style="{base_style}">' + "".join(parts) + legend_html + "</div>"

    return html

# ---- Wrapper público para notebook / pipeline ----
def visualize_spans(
    text: str,
    candidates: List[Dict[str, Any]],
    selection: Dict[str, Any],
    display_inline: bool = True,
    **render_kwargs
) -> str:
    """
    Construye segmentos y devuelve HTML.
    Si display_inline=True e IPython está disponible, hace display().
    """
    ents = collect_ents_selected_only(text, candidates, selection)
    segments = segmentize_by_boundaries(text, ents)
    segments = mask_segments_to_selected(segments, selection)

    html = render_overlaps_inline_html(text, segments, **render_kwargs)

    if display_inline:
        try:
            from IPython.display import HTML as _HTML, display as _display
            _display(_HTML(html))
        except Exception:
            pass

    return html

# =========================================================
# 8) Reporte de cierre
# =========================================================

def extract_techniques_from_text(final_answer: str) -> List[str]:
    """
    Heurística simple: extrae tokens en MAYÚSCULAS tipo técnicas.
    Ajusta regex si tus técnicas incluyen guiones/underscores.
    """
    techs = re.findall(r"\b[A-Z_]{3,}\b", final_answer or "")
    blacklist = {"TEXTO", "MODELOS", "JSON"}
    return [t for t in techs if t not in blacklist]

def synthesis_report_html(
    TEXTO: str,
    judged: Dict[str, Any],
    topk: List[Dict[str, Any]],
    synthesis: Dict[str, Any],
    selection: Optional[Dict[str, Any]] = None,   # por si quieres usarlo después
    title: str = "Síntesis"
) -> str:
    """
    Construye el HTML del reporte tipo tarjeta.
    NO hace display; solo devuelve el HTML.
    """
    final_answer = (synthesis or {}).get("final_answer", "") or ""
    brief_reasoning = (synthesis or {}).get("brief_reasoning", "") or ""

    # Técnicas: preferimos topk (más confiable); si no hay, inferimos desde final_answer
    if topk:
        techniques = []
        for c in topk:
            techniques.append({
                "model": c.get("model", ""),
                "include": bool(c.get("include", False)),
                "confidence": float(c.get("confidence", 0.0) or 0.0),
                "strengths": c.get("strengths", []) or [],
                "dependency_trace": c.get("dependency_trace", None),
            })
    else:
        techniques = [{
            "model": t,
            "include": True,
            "confidence": None,
            "strengths": [],
            "dependency_trace": None,
        } for t in extract_techniques_from_text(final_answer)]

    # Confianza global (promedio simple)
    confs = [t["confidence"] for t in techniques if t["confidence"] is not None]
    global_conf = mean(confs) if confs else None

    def conf_bar(x: Optional[float]) -> str:
        if x is None:
            return ""
        blocks = int(round(x * 10))
        blocks = max(0, min(10, blocks))
        return "█" * blocks + "░" * (10 - blocks)

    def badge(include: bool) -> str:
        return "✓" if include else "~"

    def dep_trace(dt) -> str:
        if not dt:
            return ""
        if isinstance(dt, str):
            dt = [dt]
        return f"<span class='dep'>(padre: {', '.join(map(_html.escape, dt))})</span>"

    tech_lines = []
    for t in techniques:
        tech_lines.append(
            f"""
            <div class="tech">
              <span class="badge">{badge(t["include"])}</span>
              <span class="techname">{_html.escape(t["model"])}</span>
              {"<span class='conf'>conf " + f"{t['confidence']:.2f}</span>" if t["confidence"] is not None else ""}
              {dep_trace(t.get("dependency_trace"))}
            </div>
            """
        )

    # Evidencia: strengths si existen
    ev_lines = []
    for t in techniques:
        strengths = t.get("strengths", []) or []
        for s in strengths:
            ev_lines.append(f"<li><b>{_html.escape(t['model'])}</b>: {_html.escape(str(s))}</li>")

    ev_block = "<ul>" + "\n".join(ev_lines) + "</ul>" if ev_lines else "<i>Sin evidencia extraída.</i>"

    card_html = f"""
    <style>
      .card {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px;
        background: white; box-shadow: 0 2px 6px rgba(0,0,0,.06);
        max-width: 900px; line-height: 1.35;
      }}
      .header {{display:flex; justify-content:space-between; align-items:center;}}
      .title {{font-size: 18px; font-weight: 700;}}
      .sub {{color:#6b7280; font-size: 13px; margin-top:2px;}}
      .section-title {{margin-top:12px; font-weight:700; font-size:14px; color:#111827;}}
      .final {{font-size: 14px; margin-top: 6px;}}
      .confbar {{font-family: ui-monospace, SFMono-Regular, Menlo, monospace;}}
      .tech {{display:flex; gap:8px; align-items:baseline; margin:3px 0;}}
      .badge {{display:inline-block; width:18px; text-align:center; font-weight:700;}}
      .techname {{font-weight:700;}}
      .conf {{color:#374151; font-size:13px; margin-left:6px;}}
      .dep {{color:#6b7280; font-size:12px; margin-left:6px;}}
      .reasoning {{
        background:#f9fafb; border-left:4px solid #111827; padding:8px 10px;
        border-radius:8px; margin-top:6px; font-size:14px;
      }}
    </style>

    <div class="card">
      <div class="header">
        <div>
          <div class="title">⚖️ {_html.escape(title)}</div>
          <div class="sub">Resumen del veredicto final del juez</div>
        </div>
        <div style="text-align:right">
          {f"<div class='sub'>Confianza global</div><div class='confbar'>{conf_bar(global_conf)} {global_conf:.2f}</div>" if global_conf is not None else ""}
        </div>
      </div>

      <div class="section-title">Final</div>
      <div class="final">{_html.escape(final_answer)}</div>

      <div class="section-title">Técnicas validadas</div>
      {''.join(tech_lines) if tech_lines else "<i>No hay técnicas seleccionadas.</i>"}

      <div class="section-title">Reasoning</div>
      <div class="reasoning">{_html.escape(brief_reasoning)}</div>

      <div class="section-title">Evidencia</div>
      {ev_block}
    </div>
    """
    return card_html

def synthesis_report(
    TEXTO: str,
    judged: Dict[str, Any],
    topk: List[Dict[str, Any]],
    synthesis: Dict[str, Any],
    selection: Optional[Dict[str, Any]] = None,
    title: str = "Síntesis",
    display_inline: bool = True
) -> str:
    """
    Wrapper público: devuelve HTML y opcionalmente lo muestra en Jupyter.
    """
    html_card = synthesis_report_html(
        TEXTO=TEXTO,
        judged=judged,
        topk=topk,
        synthesis=synthesis,
        selection=selection,
        title=title
    )
    if display_inline:
        try:
            from IPython.display import display as _display, HTML as _HTML
            _display(_HTML(html_card))
        except Exception:
            pass
    return html_card

def report_from_run(result: Dict[str, Any], TEXTO: str, display_inline: bool = True, title: str = "Síntesis") -> str:
    """
    Atajo: toma directamente el dict de run_juez.
    """
    return synthesis_report(
        TEXTO=TEXTO,
        judged=result.get("judged", {}),
        topk=result.get("topk", []) or [],
        synthesis=result.get("synthesis", {}),
        selection=result.get("selected_models_posthoc", None),
        title=title,
        display_inline=display_inline
    )

# =============================================================
# 9) Análisis de la aplicación de la rúbrica por técnica (juez)
# =============================================================

def rubric_table_card(
    judged: dict,
    candidates: Optional[List[Dict[str, Any]]] = None,
    title: str = "Resultados de la rúbrica"
):
    """
    Visualización tabular de la rúbrica del juez.
    - Una fila por candidate (técnica).
    - Columnas: modelo, include, total, ratio, confidence, consistencia, seguridad, preferencia_si, justificacion_breve.
    - Si 'candidates' se pasa, usa candidates[i]['model'], 'ratio', 'confidence', 'raw.justificacion_breve'.
    """

    scores_list = (judged or {}).get("scores", {}).get("candidates", [])
    if not isinstance(scores_list, list) or not scores_list:
        _display(_HTML("<i>No hay información de rúbrica en 'judged'.</i>"))
        return

    rows_html = []
    for idx, sc in enumerate(scores_list):
        sc = sc or {}
        s = sc.get("scores", {}) or {}
        include = bool(sc.get("include", False))
        total = float(sc.get("total", 0.0) or 0.0)

        # Info desde candidates, si está disponible
        cand = candidates[idx] if candidates and idx < len(candidates) and isinstance(candidates[idx], dict) else {}
        model_name = str(cand.get("model", f"IDX_{idx}"))

        ratio = cand.get("ratio", None)
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else ""

        cand_conf = cand.get("confidence", None)
        conf_str = f"{float(cand_conf):.2f}" if isinstance(cand_conf, (int, float)) else ""

        raw = cand.get("raw", {}) if isinstance(cand.get("raw"), dict) else {}
        justif_breve = raw.get("justificacion_breve") or cand.get("rationale_summary") or ""
        justif_breve_html = _html.escape(str(justif_breve))[:300]

        tr_style = "background:#ecfdf5;" if include else ""
        include_mark = "✓" if include else ""

        rows_html.append(
            f"""
            <tr style="{tr_style}">
              <td style="text-align:right; color:#6b7280;">{idx}</td>
              <td>{_html.escape(model_name)}</td>
              <td style="text-align:center;">{include_mark}</td>
              <td style="text-align:right;">{total:.2f}</td>
              <td style="text-align:right;">{ratio_str}</td>
              <td style="text-align:right;">{conf_str}</td>
              <td style="text-align:center;">{s.get('consistencia', '')}</td>
              <td style="text-align:center;">{s.get('seguridad', '')}</td>
              <td style="text-align:center;">{s.get('preferencia_si', '')}</td>
              <td style="text-align:left; max-width:320px;">{justif_breve_html}</td>
            </tr>
            """
        )

    table_html = f"""
    <style>
      .rubric-card {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        border: 1px solid #e5e7eb; border-radius: 14px;
        background: white; box-shadow: 0 2px 6px rgba(0,0,0,.06);
        max-width: 1000px; padding: 14px 16px; margin: 14px 0;
      }}
      .rubric-header {{
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom: 8px;
      }}
      .rubric-title {{
        font-size: 18px; font-weight: 700;
      }}
      .rubric-sub {{
        color:#6b7280; font-size: 13px; margin-top:2px;
      }}
      .rubric-table {{
        width:100%; border-collapse:collapse; font-size:12px; margin-top:8px;
        table-layout:fixed;
      }}
      .rubric-table th, .rubric-table td {{
        padding:4px 6px; border-bottom:1px solid #e5e7eb;
        vertical-align:top;
      }}
      .rubric-table th {{
        text-align:center; background:#f9fafb; color:#111827; font-weight:600;
      }}
      .rubric-table tr:last-child td {{
        border-bottom:none;
      }}
    </style>

    <div class="rubric-card">
      <div class="rubric-header">
        <div>
          <div class="rubric-title">📊 {_html.escape(title)}</div>
          <div class="rubric-sub">Puntuaciones por técnica, con ratio de consistencia y justificación breve.</div>
        </div>
      </div>

      <table class="rubric-table">
        <thead>
          <tr>
            <th style="width:30px;">#</th>
            <th style="text-align:left; width:120px;">Técnica / Modelo</th>
            <th style="width:30px;">✔</th>
            <th style="width:60px;">Total</th>
            <th style="width:60px;">Ratio</th>
            <th style="width:70px;">Confidence</th>
            <th style="width:80px;">Consistencia</th>
            <th style="width:80px;">Seguridad</th>
            <th style="width:90px;">Pref. Sí</th>
            <th style="text-align:left;">Justificación breve</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """

    _display(_HTML(table_html))


def show_rubric_table_from_result(
    result: dict,
    candidates: Optional[List[Dict[str, Any]]] = None,
    title: str = "Resultados de la rúbrica"
):
    """
    Conveniencia: recibe el dict completo devuelto por run_juez y
    opcionalmente la lista de candidates (para mostrar modelo, ratio, confidence, justificación breve).
    """
    judged = (result or {}).get("judged", {})
    rubric_table_card(judged=judged, candidates=candidates, title=title)
