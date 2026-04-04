# Wireless Systems Sage — Spectrum CV Extraction Backend

## Architecture
- **API Gateway** (REST) → **Lambda** (Python) → **Amazon Bedrock** (Claude Sonnet)
- The HTML frontend sends a spectrum analyzer screenshot to the API
- Lambda calls Bedrock's Converse API with vision, extracts measurement data
- Returns structured JSON back to the frontend

## Prerequisites
1. An AWS account with Bedrock access enabled
2. Claude Sonnet model enabled in Bedrock (us-east-1 or us-west-2)
3. AWS CLI configured with credentials for that account
4. Python 3.12+ and pip

## Deployment

### Option A: Quick deploy with SAM CLI
```bash
cd wireless-systems-sage
sam build
sam deploy --guided
```

### Option B: Manual deploy
1. Zip the Lambda: `cd lambda && zip -r ../function.zip . && cd ..`
2. Create Lambda function in AWS Console
3. Create API Gateway REST API pointing to the Lambda
4. Set Lambda env var `BEDROCK_REGION` (default: us-east-1)

## After Deployment
Copy the API Gateway endpoint URL and paste it into the HTML tool's
"Bedrock Proxy URL" field.
