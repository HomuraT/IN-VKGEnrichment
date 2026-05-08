"""
URI 缩短工具模块

将完整的 URI 转换为前缀格式，提升可读性。
例如: http://sws.ifi.uio.no/vocab/npd-v2#Company -> npdv:Company
"""

import re
from typing import Dict, Optional

from src.config.prefix_for_vkgs import prefix_for_vkgs


def shorten_uri(uri: str, prefix_map: Dict[str, str]) -> str:
    """
    将单个 URI 转换为前缀格式
    
    Args:
        uri: 完整的 URI，如 "http://sws.ifi.uio.no/vocab/npd-v2#Company"
        prefix_map: 前缀映射字典，如 {"npdv": "http://sws.ifi.uio.no/vocab/npd-v2#"}
    
    Returns:
        缩短后的 URI，如 "npdv:Company"；如果没有匹配的前缀，返回原 URI
    """
    if not uri or not prefix_map:
        return uri
    
    # 按命名空间长度倒序排序，优先匹配最长的前缀
    sorted_prefixes = sorted(prefix_map.items(), key=lambda x: len(x[1]), reverse=True)
    
    for prefix, namespace in sorted_prefixes:
        if uri.startswith(namespace):
            local_name = uri[len(namespace):]
            # 处理空前缀的情况
            if prefix == "" or prefix == ":":
                return f":{local_name}"
            return f"{prefix}:{local_name}"
    
    return uri


def shorten_text(text: str, prefix_map: Dict[str, str]) -> str:
    """
    批量替换文本中的所有 URI
    
    Args:
        text: 包含 URI 的文本
        prefix_map: 前缀映射字典
    
    Returns:
        替换后的文本
    """
    if not text or not prefix_map:
        return text
    
    # 匹配 URI 的正则表达式
    # 匹配 http:// 或 https:// 开头，后跟非空白字符，必须包含 # 或 /
    uri_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+(?:[#/][^\s<>"{}|\\^`\[\],\)]*)?'
    
    def replace_uri(match):
        uri = match.group(0)
        # 移除末尾可能的标点符号
        trailing_punct = ""
        while uri and uri[-1] in ".,;:!?)":
            trailing_punct = uri[-1] + trailing_punct
            uri = uri[:-1]
        
        shortened = shorten_uri(uri, prefix_map)
        return shortened + trailing_punct
    
    return re.sub(uri_pattern, replace_uri, text)


def get_prefix_map_for_vkg(vkg_name: str) -> Dict[str, str]:
    """
    从配置文件获取指定 VKG 的前缀映射
    
    Args:
        vkg_name: VKG 名称，如 "npd" 或 "bgee_v14_genex"
    
    Returns:
        前缀映射字典；如果 VKG 不存在，返回空字典
    """
    return prefix_for_vkgs.get(vkg_name, {})


def create_uri_shortener(vkg_name: Optional[str]) -> callable:
    """
    创建一个绑定了特定 VKG 前缀映射的 URI 缩短函数
    
    Args:
        vkg_name: VKG 名称
    
    Returns:
        一个接受 text 参数的函数，用于缩短文本中的 URI
    """
    if not vkg_name:
        # 如果没有指定 VKG，返回一个不做任何处理的函数
        return lambda text: text
    
    prefix_map = get_prefix_map_for_vkg(vkg_name)
    
    if not prefix_map:
        # 如果 VKG 没有配置前缀，返回一个不做任何处理的函数
        return lambda text: text
    
    # 返回一个闭包，绑定了 prefix_map
    def shortener(text: str) -> str:
        return shorten_text(text, prefix_map)
    
    return shortener

