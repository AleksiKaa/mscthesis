import re
import json


from .constants import ERROR_RESULT


def parse_output(text):
    # Try to find JSON
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return ERROR_RESULT

    try:
        data = json.loads(text[start : end + 1])
            
        # Is dict
        if not isinstance(data, dict):
            return ERROR_RESULT

        return data

    except Exception as e:
        print(e)
        return ERROR_RESULT
