from .constants import DEFAULT_RESULT

import re, json

def parse_output(text):
    # Try to find JSON
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return DEFAULT_RESULT

    try:
        data = json.loads(match.group())
    
        # Is dict
        if not isinstance(data, dict):
            return DEFAULT_RESULT
    
        return {
            "ThemeCorrect" : data.get("ThemeCorrect", DEFAULT_RESULT.get("ThemeCorrect")),
            "TopicCorrect" : data.get("TopicCorrect", DEFAULT_RESULT.get("TopicCorrect")),
            "ConceptCorrect" : data.get("ConceptCorrect", DEFAULT_RESULT.get("ConceptCorrect")),
            "Explanation": data.get("Explanation", DEFAULT_RESULT.get("Explanation"))
        }

    except Exception:
        return DEFAULT_RESULT