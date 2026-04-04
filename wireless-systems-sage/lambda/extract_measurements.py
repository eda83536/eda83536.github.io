"""
Wireless Systems Sage — Spectrum Analyzer CV Extraction
Lambda handler that calls Amazon Bedrock (Claude Sonnet) with a spectrum
analyzer screenshot and returns structured measurement data.
"""

import json
import os
import base64
import boto3
from botocore.config import Config

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(read_timeout=120, retries={"max_attempts": 2}),
)

EXTRACTION_PROMPT = """You are an expert RF measurement extraction tool used by wireless engineers.
Analyze this spectrum analyzer screenshot and extract all visible measurement data.

Return a JSON object with this structure:
{
  "confidence": <0-100 integer>,
  "instrument": { "make": "<string>", "model": "<string>", "detected": <bool> },
  "settings": {
    "center_frequency": { "value": <number>, "unit": "<Hz|kHz|MHz|GHz>" },
    "span": { "value": <number>, "unit": "<Hz|kHz|MHz|GHz>" },
    "start_frequency": { "value": <number|null>, "unit": "<Hz|kHz|MHz|GHz>" },
    "stop_frequency": { "value": <number|null>, "unit": "<Hz|kHz|MHz|GHz>" },
    "rbw": { "value": <number|null>, "unit": "<Hz|kHz|MHz>" },
    "vbw": { "value": <number|null>, "unit": "<Hz|kHz|MHz>" },
    "ref_level": { "value": <number|null>, "unit": "dBm" },
    "scale": { "value": <number|null>, "unit": "dB/div" },
    "sweep_time": { "value": <number|null>, "unit": "<ms|s>" },
    "detector_type": "<string|null>",
    "trace_mode": "<string|null>"
  },
  "markers": [
    { "id": "<M1|M2|...>", "frequency": { "value": <number>, "unit": "<MHz|GHz>" }, "power": { "value": <number>, "unit": "dBm" }, "type": "<normal|delta|peak>" }
  ],
  "peak_power": { "value": <number|null>, "unit": "dBm" },
  "noise_floor": { "value": <number|null>, "unit": "dBm", "estimation_method": "<read|estimated>" },
  "channel_power": { "value": <number|null>, "unit": "dBm", "bandwidth": { "value": <number|null>, "unit": "<MHz|kHz>" } },
  "occupied_bandwidth": { "value": <number|null>, "unit": "<MHz|kHz>" },
  "additional_measurements": {}
}

Rules:
- Return ONLY valid JSON. No markdown fences, no explanation.
- Use null for any value you cannot read or confidently determine.
- All power values in dBm. All frequencies with explicit units.
- If the image is not a spectrum analyzer screenshot, return {"error": "Not a spectrum analyzer screenshot", "confidence": 0}.
"""


def build_prompt(instrument_hint=None, focus=None):
    """Build the extraction prompt with optional hints."""
    prompt = EXTRACTION_PROMPT
    if instrument_hint and instrument_hint != "auto":
        prompt += f"\nInstrument hint: {instrument_hint}\n"
    if focus:
        prompt += f"\nFocus extraction on: {', '.join(focus)}\n"
    return prompt


def call_bedrock(image_bytes, media_type, instrument_hint=None, focus=None):
    """Call Bedrock Converse API with the image."""
    prompt = build_prompt(instrument_hint, focus)

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": media_type.split("/")[-1],  # png, jpeg, etc.
                            "source": {"bytes": image_bytes},
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 4096, "temperature": 0},
    )

    # Extract text from response
    output_text = ""
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            output_text += block["text"]

    return output_text


def parse_response(raw_text):
    """Parse the model response into JSON, handling markdown fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # remove first line
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def lambda_handler(event, context):
    """API Gateway Lambda handler."""
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": "",
        }

    try:
        body = json.loads(event.get("body", "{}"))

        # Validate input
        image_b64 = body.get("image")
        if not image_b64:
            return error_response(400, "Missing 'image' field (base64 encoded)")

        media_type = body.get("media_type", "image/png")
        instrument_hint = body.get("instrument_hint", "auto")
        focus = body.get("focus")  # optional list of focus areas

        # Decode image
        image_bytes = base64.b64decode(image_b64)

        # Size check (Bedrock limit ~20MB, but let's be reasonable)
        if len(image_bytes) > 10 * 1024 * 1024:
            return error_response(400, "Image too large. Max 10MB.")

        # Call Bedrock
        raw_text = call_bedrock(image_bytes, media_type, instrument_hint, focus)

        # Parse response
        result = parse_response(raw_text)

        return {
            "statusCode": 200,
            "headers": cors_headers(),
            "body": json.dumps(result),
        }

    except json.JSONDecodeError as e:
        return error_response(
            502, f"Model returned invalid JSON: {str(e)}"
        )
    except bedrock.exceptions.ValidationException as e:
        return error_response(400, f"Bedrock validation error: {str(e)}")
    except Exception as e:
        return error_response(500, f"Internal error: {str(e)}")


def cors_headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
    }


def error_response(status, message):
    return {
        "statusCode": status,
        "headers": cors_headers(),
        "body": json.dumps({"error": message}),
    }
