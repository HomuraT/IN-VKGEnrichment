"""
通用的 JSON 解析工具，支持自由输出 + 结构化输出降级。

优先使用 JsonOutputParser 让模型自由输出，失败后降级到结构化输出。
这样可以最大化保留模型的推理能力，同时保证输出格式的可靠性。
"""

import re
from typing import TypeVar, Type, Tuple, Dict, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.callbacks import get_usage_metadata_callback
from src.config.logging_config import get_logger
from src.tools.llm_usage_accumulator import record_call

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


def extract_json_from_markdown(text: str) -> str:
    """
    从 Markdown 代码块中提取 JSON 内容。
    
    优先提取 ```json ``` 代码块（取最后一个），如果没有则尝试提取普通 ``` ``` 代码块（取最后一个）。
    提取最后一个的原因：模型可能在分析过程中展示多个示例代码块，最后一个才是最终答案。
    
    Args:
        text: 包含代码块的文本
        
    Returns:
        提取的 JSON 字符串，如果没有代码块则返回原文本
    """
    # 优先匹配所有 ```json ... ```，取最后一个
    json_block_pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        last_match = matches[-1].strip()
        logger.debug(f"Extracted JSON from ```json``` block (found {len(matches)} blocks, using the last one)")
        return last_match
    
    # 回退到普通 ``` ... ```，取最后一个
    code_block_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        last_match = matches[-1].strip()
        logger.debug(f"Extracted JSON from ``` ``` block (found {len(matches)} blocks, using the last one)")
        return last_match
    
    # 没有代码块，返回原文本
    logger.debug("No markdown code block found, using raw text")
    return text


def invoke_with_json_fallback(
    llm: ChatOpenAI,
    prompt_text: str,
    pydantic_model: Type[T],
    operation_name: str = "llm_call",
    record_to_accumulator: bool = False,
) -> Tuple[T, Dict[str, Any], str]:
    """
    调用 LLM 并解析 JSON 输出，失败后降级到结构化输出。
    
    策略：
    1. 先让 LLM 自由输出，然后尝试提取并解析 JSON
    2. 如果解析失败，把自由输出送入结构化输出模式进行修正（不重新回答）
    
    注意：在 fallback 模式下会记录两次 LLM 调用：
    - 第一次：自由输出（标记为 operation_name.free）
    - 第二次：结构化修正（标记为 operation_name.structured_fallback）
    
    Args:
        llm: LangChain ChatOpenAI 实例
        prompt_text: 完整的 prompt 文本
        pydantic_model: Pydantic 模型类（用于解析和验证）
        operation_name: 操作名称（用于日志记录）
        record_to_accumulator: 是否自动记录到 usage accumulator（默认 False，由调用方控制）
    
    Returns:
        Tuple[T, Dict[str, Any], str]: (解析结果, usage_metadata, 原始输出文本)
        - usage_metadata: 总的 token 使用量（两次调用的总和，如果有 fallback）
        - 原始输出文本: 完整的模型输出（成功时为自由输出，fallback 时为组合输出）
    
    Raises:
        Exception: 如果结构化输出也失败，抛出异常
    """
    # 第一次尝试：自由输出 + JSON 提取
    full_output_text = ""
    usage_meta_free = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    try:
        with get_usage_metadata_callback() as cb:
            response = llm.invoke(prompt_text)
            full_output_text = response.content if hasattr(response, 'content') else str(response)
            usage_meta_free = cb.usage_metadata or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # 优先从 Markdown 代码块提取 JSON
            json_text = extract_json_from_markdown(full_output_text)
            
            # 解析 JSON
            parser = JsonOutputParser(pydantic_object=pydantic_model)
            parsed_output = parser.parse(json_text)
            result = pydantic_model(**parsed_output)
        
        logger.debug(f"[{operation_name}] JSON extraction succeeded")
        
        # 记录成功的自由输出调用
        if record_to_accumulator:
            record_call(
                operation_name,
                getattr(llm, "model", None),
                usage_meta_free,
                input_text=prompt_text,
                output_text=full_output_text
            )
        
        return result, usage_meta_free, full_output_text
    
    except Exception as e:
        logger.warning(f"[{operation_name}] JSON extraction failed: {e}, falling back to structured output")
        
        # 记录第一次失败的自由输出调用
        if record_to_accumulator:
            record_call(
                f"{operation_name}.free",
                getattr(llm, "model", None),
                usage_meta_free,
                input_text=prompt_text,
                output_text=full_output_text
            )
        
        # 第二次尝试：把失败的自由输出送入结构化模式进行修正
        # 注意：这里不重新回答问题，而是对已有的输出进行结构化提取
        try:
            # 获取 schema 定义，让模型理解目标结构
            schema_json = pydantic_model.model_json_schema()
            
            structure_prompt = f"""Below is a response that contains analysis and a JSON object, but the JSON extraction failed.
Your task: Extract and output ONLY the structured data from the original response according to the schema definition.

Target Schema:
{schema_json}

Original Response:
{full_output_text}

Please carefully extract the data from the original response and output it following the schema.
Do NOT generate new content - only extract what's already there."""

            with get_usage_metadata_callback() as cb2:
                structured_llm = llm.with_structured_output(pydantic_model)
                result = structured_llm.invoke(structure_prompt)
            
            usage_meta_structured = cb2.usage_metadata or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            logger.info(f"[{operation_name}] Structured output fallback succeeded")
            
            # 记录第二次结构化输出调用
            structured_json = result.model_dump_json(indent=2)
            if record_to_accumulator:
                record_call(
                    f"{operation_name}.structured_fallback",
                    getattr(llm, "model", None),
                    usage_meta_structured,
                    input_text=structure_prompt,
                    output_text=structured_json
                )
            
            # 组合 usage metadata（两次调用的总和）
            combined_usage = {
                "input_tokens": usage_meta_free.get("input_tokens", 0) + usage_meta_structured.get("input_tokens", 0),
                "output_tokens": usage_meta_free.get("output_tokens", 0) + usage_meta_structured.get("output_tokens", 0),
                "total_tokens": usage_meta_free.get("total_tokens", 0) + usage_meta_structured.get("total_tokens", 0),
            }
            
            # 组合输出：原始分析 + 结构化 JSON
            combined_output = f"{full_output_text}\n\n--- Structured Output (Fallback) ---\n{structured_json}"
            return result, combined_usage, combined_output
        
        except Exception as fallback_error:
            logger.error(f"[{operation_name}] Both JSON extraction and structured output failed: {fallback_error}")
            raise

