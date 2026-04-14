"""
Pre-translates Spanish docstrings from propaganda_pipeline.py to English.
Saves results to translations_cache.json for use during Sphinx builds.
Run once before building docs: python docs_src/translate_docstrings.py
"""
import ast
import json
import re
import time
from pathlib import Path

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

SOURCE = Path(__file__).parent.parent / "propaganda_pipeline.py"
CACHE = Path(__file__).parent / "translations_cache.json"

MAX_CHUNK = 4500  # Google Translate limit

# Patterns that indicate Spanish content even in otherwise-English docstrings
SPANISH_PATTERN = re.compile(
    r'[áéíóúñÁÉÍÓÚÑ¿¡]'
    r'|(?<!\w)(Sí|No aplica|citas|frases|razones|etiqueta|técnica'
    r'|salida|entrada|justificacion|puntaje|marcadores|confianza'
    r'|rasgos|apelaciones|presencia|objetivo)(?!\w)',
    re.IGNORECASE,
)


def has_spanish(text: str) -> bool:
    """True if docstring is primarily Spanish OR contains significant Spanish fragments."""
    try:
        sample = " ".join(text.split()[:80])
        if detect(sample) == "es":
            return True
    except LangDetectException:
        pass
    return bool(SPANISH_PATTERN.search(text))


def translate_text(text: str) -> str:
    translator = GoogleTranslator(source="auto", target="en")
    if len(text) <= MAX_CHUNK:
        return translator.translate(text)
    # Split into lines, batch into chunks under limit
    lines = text.split("\n")
    chunks, current, current_len = [], [], 0
    for line in lines:
        if current_len + len(line) + 1 > MAX_CHUNK and current:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(translator.translate(chunk))
        time.sleep(0.3)
    return "\n".join(translated_chunks)


def extract_docstrings(source_path: Path) -> dict[str, str]:
    """Return {name: docstring} for all nodes with Spanish (or mixed) docstrings.

    Uses the node's simple name as key; duplicate names (e.g. multiple
    ``postprocess`` methods) are assumed to share the same docstring.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    results = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if docstring and has_spanish(docstring):
            results[node.name] = docstring
    return results


def main():
    print("Extracting docstrings with Spanish content...")
    docstrings = extract_docstrings(SOURCE)
    print(f"Found {len(docstrings)} docstrings to translate.")

    cache: dict[str, str] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached translations.")

    for i, (name, doc) in enumerate(docstrings.items(), 1):
        if name in cache:
            print(f"  [{i}/{len(docstrings)}] {name} — cached, skipping")
            continue
        print(f"  [{i}/{len(docstrings)}] {name} — translating ({len(doc)} chars)...")
        try:
            translated = translate_text(doc)
            cache[name] = translated
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.5)
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone. {len(cache)} translations saved to {CACHE}")


if __name__ == "__main__":
    main()
