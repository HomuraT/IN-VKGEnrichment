"""
向量数据库服务的 Pydantic 数据模型

定义API请求和响应的数据结构
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """检索请求模型"""
    query: str = Field(..., description="查询文本")
    k: int = Field(default=5, ge=1, le=100, description="返回文档数量")
    with_score: bool = Field(default=False, description="是否返回相似度分数")


class DocumentModel(BaseModel):
    """文档模型"""
    page_content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    score: Optional[float] = Field(default=None, description="相似度分数（可选）")


class SearchResponse(BaseModel):
    """检索响应模型"""
    documents: List[DocumentModel] = Field(default_factory=list, description="检索到的文档列表")
    query: str = Field(..., description="原始查询文本")
    k: int = Field(..., description="请求返回的文档数量")
    actual_count: int = Field(..., description="实际返回的文档数量")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(default="ok", description="服务状态")
    db_name: str = Field(..., description="向量库名称")
    db_path: str = Field(..., description="向量库路径")


class InfoResponse(BaseModel):
    """信息响应模型"""
    db_name: str = Field(..., description="向量库名称")
    db_path: str = Field(..., description="向量库路径")
    collection_name: str = Field(..., description="集合名称")
    embedding_model: str = Field(..., description="嵌入模型")
    document_count: Any = Field(..., description="文档数量")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="其他信息")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误信息")

