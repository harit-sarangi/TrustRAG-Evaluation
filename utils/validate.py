import sys
import json
import jsonlines
import jsonschema
from jsonschema import validate
from loguru import logger

QUESTION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Question schema",
    "type": "object",
    "properties": {"id": {"type": "integer", "description": "Question ID"}, "question": {"type": "string", "description": "The question"}},
    "required": ["id", "question"],
}

ANSWER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Answer file schema",
    "type": "object",
    "properties": {
        "id": {"type": "integer", "description": "Question ID"},
        "question": {"type": "string", "description": "The question"},
        "passages": {
            "type": "array",
            "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance",
            "items": {
                "type": "object",
                "properties": {
                    "passage": {"type": "string", "description": "Passage text"},
                    "doc_IDs": {
                        "type": "array",
                        "description": "Passage related FineWeb doc IDs, ordered by decreasing importance",
                        "items": {"type": "string", "description": "FineWeb doc ID, e.g., <urn:uuid:d69cbebc-133a-4ebe-9378-68235ec9f091>"},
                    },
                },
                "required": ["passage", "doc_IDs"],
            },
        },
        "final_prompt": {"type": "string", "description": "Final prompt, as submitted to Falcon LLM"},
        "answer": {"type": "string", "description": "Your answer"},
    },
    "required": ["id", "question", "passages", "final_prompt", "answer"],
}


def read_json_schema(schema_file: str = "answer-schema.json") -> dict:
    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Schema file {schema_file} not found.")
        return None


def validate_json(json_file: str, schema: dict, item_type: str) -> bool:
    """
    Validate items in a JSONL file against a schema.

    Args:
        json_file: Path to the JSONL file
        schema: JSON schema to validate against, the schema should contain "id" field
        item_type: Type of items being validated (for logging purposes, e.g., "Question" or "Answer")

    Returns:
        bool: True if errors were found, False otherwise
    """
    has_errors = False
    try:
        if "id" not in schema.get("properties", {}) or "id" not in schema.get("required", []):
            logger.error("Schema must include 'id' in properties and required fields.")
            return True

        total_ids, unique_ids = 0, set()
        with jsonlines.open(json_file) as reader:
            for item in reader:
                try:
                    validate(instance=item, schema=schema)
                    logger.trace(f"{item_type} {item['id']} is valid.")
                    unique_ids.add(item["id"])
                    total_ids += 1
                except jsonschema.exceptions.ValidationError as e:
                    logger.error(f"{item_type} {item['id']} is invalid: {e.message}")
                    has_errors = True

        if total_ids != len(unique_ids):
            logger.warning(f"Number of unique IDs ({len(unique_ids)}) does not match total IDs ({total_ids})!")

        logger.info(f"Verified {len(unique_ids)} unique {item_type.lower()}s.")
    except Exception as e:
        logger.error(f"Error reading or processing {json_file}: {str(e)}")
        return True
    return has_errors


def validate_questions(json_file: str = "questions.jsonl", schema: dict = QUESTION_SCHEMA) -> bool:
    return validate_json(json_file, schema, "Question")


def validate_answers(json_file: str = "answers.jsonl", schema: dict = ANSWER_SCHEMA) -> bool:
    return validate_json(json_file, schema, "Answer")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=0)

    # Get filename from command line argument if provided, otherwise use default
    json_file = sys.argv[1] if len(sys.argv) > 1 else "answers.jsonl"

    has_errors = validate_answers(json_file)
    sys.exit(1 if has_errors else 0)
