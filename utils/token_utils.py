"""
Token utilities using tiktoken for estimation and context management.

This module provides:
- Encoding selection per provider/model
- Token counting for text and messages
- Reconciliation of estimated vs actual token usage
- Context-fit guards with summarize/truncate strategies
"""

from nt import system
from unittest import result
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
    if context_strs:
        for ctx in context_strs:
            context_tokens += len(enc.encode(ctx, disallowed_special=()))

    overhead = 3

    return {
            "input_tokens" : input_tokens,
            "context_tokens": context_tokens,
            "estimated_total": input_tokens + context_tokens + overhead
            }    


def reconcile_usage(estimate, provider_usage = None):
    """
    Merge estimated vs actual token usage from provider.

    Args:
        estimate: Dict with input_tokens, context_tokens, estimated_total
        provider_usage: Optional usage dict from provider API response

    Returns:
        Dict with both estimated and actual fields
    """

    result = {
            "input_tokens_est": estimate.get("input_tokens", 0),
            "context_tokens_est": estimate.get("context_tokens", 0),
            "total_est": estimate.get("estimated_total", 0),
            "prompt_tokens_actual":None,
            "completion_tokens_actual": None,
            "total_tokens_actual": None
            }
    
    if provider_usage:
        # Openai format
        if "prompt_tokens" in provider_usage:
            result["prompt_tokens_actual"] = provider_usage["prompt_tokens"]
            result["completion_tokens_actual"] = provider_usage.get("completion_tokens", 0)
            result["total_tokens_actual"] = provider_usage.get("total_tokens", 0)
        # Google gemini format
        elif "promptTokenCount" in provider_usage:
            result["prompt_tokens_actual"] = provider_usage.get("promptTokenCount") or 0
            result["completion_tokens_actual"] = provider_usage.get("candidatesTokenCount") or 0
            result["total_tokens_actual"] = (
                result["prompt_tokens_actual"] + result["completion_tokens_actual"]
            )
    
        return result


def estimate_prompt_tokens(
                            messages,
                            provider,
                            model,
                            context_strs = None
                            ):

    counts = count_message_tokens(messages, provider, model, context_strs)
    return counts["estimated_total"]


def fit_within_context(
                        messages,
                        provider,
                        model,
                        max_context_tokens,
                        strategy,
                        context_strs = None
                        ):

    """
    Ensure messages + context fit within token budget.

    Strategies:
    - summarize: Add a summarization step to condense content
    - truncate: Remove oldest messages until under budget

    Args:
        messages: Messages array
        provider: API provider
        model: Model identifier
        max_context_tokens: Maximum allowed prompt tokens
        strategy: How to handle overflow
        context_strs: Optional context strings

    Returns:
        Tuple of (adjusted_messages, adjusted_context, metadata_dict)
    """
    current_tokens = estimate_prompt_tokens(messages, provider, model, context_strs)

    if current_tokens <= max_context_tokens:
        return messages, context_strs, {"overflow": False, "original_tokens": current_tokens}
    
    # Overflow detected
    if strategy == "truncate":
        # Remove oldest messages
        adjusted = messages.copy()
        system_msgs = [m for m in adjusted if m.get("role") == "system"]
        other_msgs = [m for m in adjusted if m.get("role") != "system"]

        # Truncate
        while len(other_msgs) > 1 and estimate_prompt_tokens( 
            system_msgs + other_msgs, provider, model, context_strs
        ) > max_context_tokens:
            other_msgs.pop(0)
        
        # If still overbudget
        if estimate_prompt_tokens(
            system_msgs + other_msgs, provider, model, context_strs
        ) > max_context_tokens and other_msgs:
            # Truncate the last message's content to fit
            last_msg = other_msgs[-1]
            enc = pick_encoding(provider, model)

            # Calculate available tokens for content
            overhead = 4 * (len(system_msgs) + 1) + 3
            available_tokens = max_context_tokens - overhead

            # Encode and truncate
            content = last_msg.get("content", "")
            tokens = enc.encode(content, disallowed_special=())
            if len(tokens) > available_tokens:
                tokens = tokens[:available_tokens]
                truncated_content = enc.decode(tokens)
                other_msgs[-1] = {**last_msg, "content": truncated_content + "... [truncated]"}

        return (
                system_msgs + other_msgs,
                context_strs,
                {
                    "overflow": True,
                    "original_tokens": current_tokens,
                    "strategy": "truncate",
                    "messages_removed": len(messages) - len(system_msgs) - len(other_msgs),                    
                }
                )

    elif strategy == "summarize":
        # Return instruction to summarize
        return (
            messages,
            context_strs,
            {
                "overflow": True,
                "original_tokens": current_tokens,
                "strategy": "summarize",
                "action_required": "Use overflow_summarize prompt",
            },
        )

    return messages, context_strs, {"overflow": False}

