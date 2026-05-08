"""
通用向量数据库服务

FastAPI 应用，提供向量库检索的 HTTP API。
每个服务实例加载一个向量库，通过命令行参数指定。
"""

import argparse
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from src.services.vector_db.local_retriever import LocalVectorRetriever
from src.services.vector_db.models import (
    SearchRequest,
    SearchResponse,
    DocumentModel,
    HealthResponse,
    InfoResponse,
    ErrorResponse,
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# 全局变量：向量库检索器
retriever: Optional[LocalVectorRetriever] = None
db_name: str = ""
db_path: str = ""
collection_name: str = "default"
embedding_model: str = ""


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="Vector Database Service",
        description="向量数据库检索服务",
        version="1.0.0",
    )
    
    @app.get("/", response_model=dict)
    async def root():
        """根路径"""
        return {
            "service": "Vector Database Service",
            "db_name": db_name,
            "endpoints": {
                "health": "/health",
                "info": "/info",
                "search": "/search (POST)",
            }
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """健康检查"""
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        return HealthResponse(
            status="ok",
            db_name=db_name,
            db_path=db_path,
        )
    
    @app.get("/info", response_model=InfoResponse)
    async def info():
        """获取向量库信息"""
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        info_dict = retriever.get_info()
        return InfoResponse(
            db_name=db_name,
            db_path=db_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            document_count=info_dict.get("document_count", "unknown"),
            additional_info=info_dict,
        )
    
    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """
        向量检索接口
        
        Args:
            request: 检索请求，包含查询文本和返回数量
        
        Returns:
            检索结果
        """
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        try:
            if request.with_score:
                # 返回带分数的结果
                results_with_score = retriever.similarity_search_with_score(
                    query=request.query,
                    k=request.k
                )
                documents = [
                    DocumentModel(
                        page_content=doc.page_content,
                        metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                        score=float(score) if score is not None else None,
                    )
                    for doc, score in results_with_score
                ]
            else:
                # 返回不带分数的结果
                results = retriever.similarity_search(
                    query=request.query,
                    k=request.k
                )
                documents = [
                    DocumentModel(
                        page_content=doc.page_content,
                        metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                    )
                    for doc in results
                ]
            
            return SearchResponse(
                documents=documents,
                query=request.query,
                k=request.k,
                actual_count=len(documents),
            )
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """全局异常处理"""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc)
            ).model_dump()
        )
    
    return app


def initialize_retriever(
    db_name_arg: str,
    base_dir: str,
    collection_name_arg: str,
    embedding_model_arg: str,
):
    """
    初始化向量库检索器
    
    Args:
        db_name_arg: 向量库名称（不含.chroma后缀）
        base_dir: 向量库基础目录
        collection_name_arg: 集合名称
        embedding_model_arg: 嵌入模型键名
    """
    global retriever, db_name, db_path, collection_name, embedding_model
    
    db_name = db_name_arg
    collection_name = collection_name_arg
    embedding_model = embedding_model_arg
    
    # 构建完整路径
    db_path = os.path.join(base_dir, f"{db_name}.chroma")
    
    logger.info(f"Initializing vector database service...")
    logger.info(f"  DB Name: {db_name}")
    logger.info(f"  DB Path: {db_path}")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Embedding Model: {embedding_model}")
    
    # 检查路径是否存在
    if not os.path.exists(db_path):
        logger.error(f"Vector database not found: {db_path}")
        raise FileNotFoundError(f"Vector database not found: {db_path}")
    
    # 初始化检索器
    try:
        retriever = LocalVectorRetriever(
            vector_db_path=db_path,
            collection_name=collection_name,
            embedding_model_key=embedding_model,
        )
        logger.success(f"✅ Vector database service initialized successfully!")
        
        # 显示向量库信息（跳过文档数量统计，避免大型向量库启动缓慢）
        logger.info(f"  Skipping document count for faster startup (use /info endpoint to get count)")
    
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="向量数据库检索服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-name",
        type=str,
        required=True,
        help="向量库名称（不含.chroma后缀），例如：bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_ontology_elements",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口号",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="resources/vector_databases",
        help="向量库基础目录",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="集合名称",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="local_qwen_3_8b_embedding",
        help="嵌入模型键名",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker 数量",
    )
    
    args = parser.parse_args()
    
    # 初始化检索器
    try:
        initialize_retriever(
            db_name_arg=args.db_name,
            base_dir=args.base_dir,
            collection_name_arg=args.collection,
            embedding_model_arg=args.embedding_model,
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    # 创建并启动应用
    app = create_app()
    
    logger.info(f"🚀 Starting service on {args.host}:{args.port}")
    logger.info(f"📖 API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()

