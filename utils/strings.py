from loguru import logger


def clean_str(s):
    try:
        if isinstance(s, dict):
            s = " ".join([f"{k}: {v}" for k, v in s.items()])
        elif isinstance(s, list):
            s = " ".join(map(str, s))
        else:
            s = str(s)
    except Exception as e:
        logger.opt(exception=e).error("the output cannot be converted to a string")
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()
