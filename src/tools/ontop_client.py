import httpx
import os
from typing import Dict, Any, Optional, Union
import json
import pandas as pd
from src.tools import sparql_cache as _sparql_cache


class OntopClient:
    """
    Ontop SPARQL 端点客户端
    
    提供简单的 SPARQL 查询执行功能，通过 HTTP 协议与 Ontop 端点通信
    """
    
    def __init__(self, endpoint_url: str = "http://localhost:8080/sparql", timeout: int = 30):
        """
        初始化 Ontop 客户端
        
        Args:
            endpoint_url (str): Ontop SPARQL 端点 URL，默认为 "http://localhost:8080/sparql"
            timeout (int): 请求超时时间（秒），默认为 30
        """
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        # 持久 HTTP 客户端，复用连接（keep-alive）并启用 HTTP/2 多路复用
        self._client: httpx.Client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
            http2=True,
        )
        
    def _http(self) -> httpx.Client:
        """Get the persistent httpx client (keep-alive/HTTP2)."""
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            self._client.close()
        except Exception:
            pass

    def execute_sparql(self, sparql_query: str, accept_format: str = "application/json") -> Union[Dict[str, Any], str]:
        """
        执行 SPARQL 查询
        
        Args:
            sparql_query (str): 要执行的 SPARQL 查询语句
            accept_format (str): 期望的返回格式，默认为 "application/json"
            
        Returns:
            Union[Dict[str, Any], str]: 若 Content-Type 为 JSON 则返回 JSON 对象；否则返回原始字符串（如 Turtle）
            
        Raises:
            requests.RequestException: 网络请求异常
            json.JSONDecodeError: JSON 解析异常
        """
        # 准备请求头
        headers = {
            'Content-Type': 'application/sparql-query',
            'Accept': accept_format
        }
        
        # 发送 POST 请求（复用连接）
        client = self._http()
        response = client.post(
            self.endpoint_url,
            headers=headers,
            content=sparql_query,
        )
        # 检查 HTTP 状态码（不吞错）
        response.raise_for_status()
        # 按 Content-Type 选择解析方式
        content_type = str(response.headers.get('content-type') or '').lower()
        if 'json' in content_type:
            return response.json()
        return response.text
    
    def execute_sparql_with_get(self, sparql_query: str, accept_format: str = "application/json") -> Union[Dict[str, Any], str]:
        """
        使用 GET 方法执行 SPARQL 查询
        
        Args:
            sparql_query (str): 要执行的 SPARQL 查询语句
            accept_format (str): 期望的返回格式，默认为 "application/json"
            
        Returns:
            Union[Dict[str, Any], str]: 若 Content-Type 为 JSON 则返回 JSON 对象；否则返回原始字符串（如 Turtle）
            
        Raises:
            requests.RequestException: 网络请求异常
            json.JSONDecodeError: JSON 解析异常
        """
        # 准备请求参数
        params = {
            'query': sparql_query
        }
        
        headers = {
            'Accept': accept_format
        }
        
        # 发送 GET 请求（复用连接）
        client = self._http()
        response = client.get(
            self.endpoint_url,
            headers=headers,
            params=params,
        )
        # 检查 HTTP 状态码
        response.raise_for_status()
        # 按 Content-Type 选择解析方式
        content_type = str(response.headers.get('content-type') or '').lower()
        if 'json' in content_type:
            return response.json()
        return response.text

    def _default_accept_header_for_mixed(self) -> str:
        # 优先 JSON 结果集，其次 Turtle/JSON-LD/RDF/XML，最后纯文本
        return (
            "application/sparql-results+json, "
            "text/turtle;q=0.9, "
            "application/ld+json;q=0.8, "
            "application/rdf+xml;q=0.7, "
            "text/plain;q=0.6, */*;q=0.1"
        )

    def _is_sparql_json_resultset(self, obj: Any) -> bool:
        return isinstance(obj, dict) and isinstance(obj.get("head"), dict) and isinstance(obj.get("results"), dict)

    def execute_sparql_mixed(self, sparql_query: str, accept_header: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        执行 SPARQL（POST），若为结果集（带 head/results）则解析为 DataFrame，否则返回字符串（如 Turtle/JSON-LD 等）。
        """
        headers = {
            'Content-Type': 'application/sparql-query',
            'Accept': (accept_header or self._default_accept_header_for_mixed()),
        }
        client = self._http()
        response = client.post(
            self.endpoint_url,
            headers=headers,
            content=sparql_query,
        )
        response.raise_for_status()
        content_type = str(response.headers.get('content-type') or '').lower()
        if 'json' in content_type:
            obj = response.json()
            if self._is_sparql_json_resultset(obj):
                return self.sparql_to_dataframe(obj)
            # 非结果集 JSON（如 JSON-LD），统一返回字符串，保持直观
            return response.text
        # 其他类型（turtle/rdf+xml/plain 等）统一返回字符串
        return response.text

    def execute_sparql_with_get_mixed(self, sparql_query: str, accept_header: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        执行 SPARQL（GET），若为结果集（带 head/results）则解析为 DataFrame，否则返回字符串。
        """
        params = {'query': sparql_query}
        headers = {'Accept': (accept_header or self._default_accept_header_for_mixed())}
        client = self._http()
        response = client.get(
            self.endpoint_url,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        content_type = str(response.headers.get('content-type') or '').lower()
        if 'json' in content_type:
            obj = response.json()
            if self._is_sparql_json_resultset(obj):
                return self.sparql_to_dataframe(obj)
            return response.text
        return response.text
    
    def set_endpoint_url(self, new_url: str) -> None:
        """
        更新端点 URL
        
        Args:
            new_url (str): 新的端点 URL
        """
        self.endpoint_url = new_url
    
    def set_timeout(self, new_timeout: int) -> None:
        """
        更新超时时间
        
        Args:
            new_timeout (int): 新的超时时间（秒）
        """
        self.timeout = new_timeout
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """
        获取当前端点配置信息
        
        Returns:
            Dict[str, Any]: 端点配置信息
        """
        return {
            'endpoint_url': self.endpoint_url,
            'timeout': self.timeout
        }
    
    def test_connection(self) -> bool:
        """
        测试与 Ontop 端点的连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 执行一个简单的查询来测试连接
            test_query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
            self.execute_sparql(test_query)
            return True
        except Exception:
            return False
    
    def sparql_to_dataframe(self, sparql_result: Dict[str, Any]) -> pd.DataFrame:
        """
        将 SPARQL 查询结果转换为 pandas DataFrame
        
        支持 SELECT 和 ASK 查询类型：
        - SELECT: 返回多列 DataFrame
        - ASK: 返回单列单行 DataFrame，列名为 'boolean'
        
        Args:
            sparql_result (Dict[str, Any]): SPARQL 查询的 JSON 结果
            
        Returns:
            pd.DataFrame: 转换后的 DataFrame
            
        Raises:
            KeyError: 结果格式不正确
            ValueError: 数据格式错误
        """
        # 主动检测结果结构，缺失时返回空表（保留列信息若可得），避免吞错
        if not isinstance(sparql_result, dict):
            return pd.DataFrame()
        
        # Handle ASK queries: {"head": {}, "boolean": true/false}
        if 'boolean' in sparql_result:
            boolean_value = sparql_result['boolean']
            return pd.DataFrame({'boolean': [boolean_value]})
        
        results_obj = sparql_result.get('results')
        if not isinstance(results_obj, dict) or 'bindings' not in results_obj:
            # 若 head.vars 存在，返回空列表，便于上层保持一致处理
            head_obj = sparql_result.get('head')
            if isinstance(head_obj, dict) and isinstance(head_obj.get('vars'), list):
                return pd.DataFrame(columns=list(head_obj.get('vars') or []))
            return pd.DataFrame()

        bindings = results_obj['bindings']
        
        if not bindings:
            # 如果没有结果，返回空的 DataFrame
            if 'head' in sparql_result and 'vars' in sparql_result['head']:
                columns = sparql_result['head']['vars']
                return pd.DataFrame(columns=columns)
            else:
                return pd.DataFrame()
        
        # 提取列名
        if 'head' in sparql_result and isinstance(sparql_result['head'], dict) and 'vars' in sparql_result['head']:
            columns = sparql_result['head']['vars']
        else:
            # 从第一行数据中提取列名
            columns = list(bindings[0].keys())
        
        # 提取数据
        data = []
        for binding in bindings:
            row = {}
            for col in columns:
                if col in binding:
                    # 提取值，SPARQL 结果中每个值都有 type 和 value 字段
                    if isinstance(binding[col], dict) and 'value' in binding[col]:
                        row[col] = binding[col]['value']
                    else:
                        row[col] = binding[col]
                else:
                    row[col] = None
            data.append(row)
        
        return pd.DataFrame(data, columns=columns)
    
    def execute_sparql_to_dataframe(self, sparql_query: str, accept_format: str = "application/json") -> pd.DataFrame:
        """
        执行 SPARQL 查询并返回 pandas DataFrame
        
        Args:
            sparql_query (str): 要执行的 SPARQL 查询语句
            accept_format (str): 期望的返回格式，默认为 "application/json"
            
        Returns:
            pd.DataFrame: 查询结果的 DataFrame 格式
            
        Raises:
            requests.RequestException: 网络请求异常
            json.JSONDecodeError: JSON 解析异常
            KeyError: 结果格式不正确
            ValueError: 数据格式错误
        """
        # 1) 缓存命中（VKG + SPARQL），resources/cache 目录
        base_dir = os.path.join("resources", "cache")
        os.makedirs(base_dir, exist_ok=True)
        key = _sparql_cache.cache_key(getattr(self, "vkg_name", None), sparql_query)
        path = _sparql_cache.cache_path(base_dir, getattr(self, "vkg_name", None), key)
        cached = _sparql_cache.load_df(path)
        if isinstance(cached, pd.DataFrame):
            return cached

        # 2) 未命中则执行
        try:
            result = self.execute_sparql(sparql_query, accept_format)
        except httpx.HTTPStatusError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 400:
                df = pd.DataFrame()
                _sparql_cache.save_df(path, df)
                return df
            raise
        df = self.sparql_to_dataframe(result)
        if isinstance(df, pd.DataFrame):
            _sparql_cache.save_df(path, df)
        return df
    
    def execute_sparql_with_get_to_dataframe(self, sparql_query: str, accept_format: str = "application/json") -> pd.DataFrame:
        """
        使用 GET 方法执行 SPARQL 查询并返回 pandas DataFrame
        
        Args:
            sparql_query (str): 要执行的 SPARQL 查询语句
            accept_format (str): 期望的返回格式，默认为 "application/json"
            
        Returns:
            pd.DataFrame: 查询结果的 DataFrame 格式
            
        Raises:
            requests.RequestException: 网络请求异常
            json.JSONDecodeError: JSON 解析异常
            KeyError: 结果格式不正确
            ValueError: 数据格式错误
        """
        base_dir = os.path.join("resources", "cache")
        os.makedirs(base_dir, exist_ok=True)
        key = _sparql_cache.cache_key(getattr(self, "vkg_name", None), sparql_query)
        path = _sparql_cache.cache_path(base_dir, getattr(self, "vkg_name", None), key)
        cached = _sparql_cache.load_df(path)
        if isinstance(cached, pd.DataFrame):
            return cached
        try:
            result = self.execute_sparql_with_get(sparql_query, accept_format)
        except httpx.HTTPStatusError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 400:
                df = pd.DataFrame()
                _sparql_cache.save_df(path, df)
                return df
            raise
        df = self.sparql_to_dataframe(result)
        if isinstance(df, pd.DataFrame):
            _sparql_cache.save_df(path, df)
        return df


if __name__ == "__main__":
    """
    测试 Ontop Client 功能
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Ontop Client 测试")
    parser.add_argument("--endpoint", "-e", default="http://localhost:8080/sparql", help="Ontop SPARQL 端点 URL")
    parser.add_argument("--query", "-q", help="要执行的 SPARQL 查询")
    parser.add_argument("--method", "-m", choices=["post", "get"], default="post", help="HTTP 请求方法")
    parser.add_argument("--test-connection", "-t", action="store_true", help="测试连接")
    parser.add_argument("--dataframe", "-d", action="store_true", help="以 DataFrame 格式返回结果")
    
    args = parser.parse_args()
    
    # 初始化 Ontop Client
    print("初始化 Ontop Client...")
    client = OntopClient(endpoint_url=args.endpoint)
    print(f"✅ Ontop Client 已初始化")
    print(f"端点 URL: {args.endpoint}")
    
    # 测试连接
    if args.test_connection:
        print("\n测试连接...")
        is_connected = client.test_connection()
        if is_connected:
            print("✅ 连接成功")
        else:
            print("❌ 连接失败")
            exit(1)
    
    # 默认测试查询（包含 SELECT / ASK / CONSTRUCT / DESCRIBE）
    if not args.query and not args.test_connection:
        print("\n执行默认测试查询（包含 SELECT/ASK/CONSTRUCT/DESCRIBE）...")
        test_cases = [
            ("SELECT", "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"),
            ("ASK", "ASK WHERE { ?s ?p ?o }"),
            ("CONSTRUCT", "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 5"),
            ("DESCRIBE", "DESCRIBE ?s WHERE { ?s ?p ?o } LIMIT 10"),
        ]

        for i, (qtype, query) in enumerate(test_cases, 1):
            print(f"\n=== 测试查询 {i} / {qtype} ===")
            print(f"查询: {query}")

            try:
                # 1) 原始返回（按 --method 使用 POST/GET）
                print("\n--- 原始返回 ---")
                if args.method == "post":
                    raw = client.execute_sparql(query)
                else:
                    raw = client.execute_sparql_with_get(query)

                if isinstance(raw, dict):
                    print(json.dumps(raw, indent=2, ensure_ascii=False))
                else:
                    preview = raw if len(raw) <= 1000 else (raw[:1000] + "\n... (truncated) ...")
                    print(preview)

                # 2) 混合返回：SELECT → DataFrame，其它类型多为文本（Turtle/JSON-LD/JSON 等）
                print("\n--- 混合返回（自动判别结果集→DataFrame，否则文本） ---")
                if args.method == "post":
                    mixed = client.execute_sparql_mixed(query)
                else:
                    mixed = client.execute_sparql_with_get_mixed(query)

                if isinstance(mixed, pd.DataFrame):
                    print(mixed)
                    print(f"数据形状: {mixed.shape}")
                    if not mixed.empty:
                        print(f"列名: {list(mixed.columns)}")
                else:
                    preview_m = mixed if len(mixed) <= 1000 else (mixed[:1000] + "\n... (truncated) ...")
                    print(preview_m)

                # 3) 仅对 SELECT 再演示 DataFrame 专用接口
                if qtype == "SELECT":
                    print("\n--- DataFrame 专用接口（仅 SELECT） ---")
                    if args.method == "post":
                        df = client.execute_sparql_to_dataframe(query)
                    else:
                        df = client.execute_sparql_with_get_to_dataframe(query)
                    print(df)
                    print(f"数据形状: {df.shape}")
                    if not df.empty:
                        print(f"列名: {list(df.columns)}")

            except Exception as e:
                print(f"❌ 查询执行失败: {e}")
