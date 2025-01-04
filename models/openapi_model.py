# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from openai import APIConnectionError, RateLimitError, OpenAI

from models.utils import trim_predictions_to_max_token_length

######################################################################################################
######################################################################################################
###
### Please pay special attention to the comments that start with "TUNE THIS VARIABLE"
###                        as they depend on your model and the available GPU resources.
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


#### CONFIG PARAMETERS ---

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
BATCH_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

#### CONFIG PARAMETERS END---

# load .env file
from dotenv import load_dotenv
load_dotenv()

class OpenAIModel:
    def __init__(self):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        # self.initialize_models()
        self.openai_client = OpenAI()
        # self.model_name = "gpt-3.5-turbo"
        self.model_name = "gpt-4o-mini"

    def call_llm_generate(self, messages, max_retries=10):
        # todo: add default response when all efforts fail
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except (APIConnectionError, RateLimitError):
                logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        return None