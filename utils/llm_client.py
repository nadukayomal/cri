import os
import time
import random
from openai import OpenAI, OpenAIError
from google import genai
from google.genai import types
from groq import Groq
from .token_utils import (
                        count_messages_tokens,
                        reconcile_usage,
                        fit_within_context,
                        )
from .router import get_context_window
from dotenv import load_dotenv
from .config_loader import (
                            get_max_retries, 
                            get_backoff_base, 
                            get_backoff_jitter
                            )


class LLMClient:
    """
    Unified client for multiple LLM providers with robust error handling.

    Features:
    - Automatic token estimation and context overflow handling
    - Retry logic with exponential backoff + jitter
    - Usage tracking (estimated vs actual)
    - Consistent return format across providers

    """

    def __init__(
                    self,
                    provider,
                    model,
                    max_retries = None,
                    backoff_base = None,
                    backoff_jitter = None,
                    hard_prompt_cap = None
                ):

        self.provider = provider
        self.model = model
        self.max_retries = max_retries if max_retries is not None else get_max_retries()
        self.backoff_base = backoff_base if backoff_base is not None else get_backoff_base()
        self.backoff_jitter = backoff_jitter if backoff_jitter is not None else get_backoff_jitter()
        self.hard_prompt_cap = hard_prompt_cap

        self._init_client()

    
