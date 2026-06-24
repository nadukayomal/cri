"""
Token utilities using tiktoken for estimation and context management.

This module provides:
- Encoding selection per provider/model
- Token counting for text and messages
- Reconciliation of estimated vs actual token usage
- Context-fit guards with summarize/truncate strategies
"""

from openai.resources.responses.input_tokens import input_token_count_params
from openai.resources.skills.skills import Content
import tiktoken

def pick_encoding(provider, model):
    """
    Select appropriate tiktoken encoding for provider/model.

    OpenAI: Use o200k_base for 4.x/o3 models, cl100k_base as fallback
    Google/Groq: Use o200k_base as approximation (caveat: not exact)

    Args:
        provider: API provider name
        model: Model identifier

    Returns:
        tiktoken.Encoding instance
    """
    if provider == "openai":
        if any(x in model.lower() for x in ["gpt-4o", "gpt-4", "o3", "o1"]):
            try:
                return tiktoken.get_encoding("o200k_base") # For GPT-4o, GPT-4, o3 models, prefer o200k_base
            except:
                pass
        return tiktoken.get_encoding("cl100k_base")   # cl100k_base for GPT-3.5 and older

    # For non-OpenAI providers, use o200k_base as approximation
    return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text,  provider,   model):
    if not text:
        return 0
    enc = pick_encoding(provider, model)
    return len(enc.encode(text, disallowed_special=()))


def count_message_tokens(
                        messages,
                        provider,
                        model,
                        context_strs = None
                        ):
    """
    Count tokens in a messages array, separating input vs context.

    Input tokens: system + user messages
    Context tokens: additional context strings (e.g., RAG documents)
    Estimated total: input + context + overhead

    Args:
        messages: OpenAI-style messages array
        provider: API provider
        model: Model identifier
        context_strs: Optional list of context strings to count separately

    Returns:
        Dict with input_tokens, context_tokens, estimated_total
    """               
    enc = pick_encoding(provider, model)

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        input_tokens += 4
        input_tokens += len(enc.encode(content, disallowed_special=()))

    context_tokens = 0
    
