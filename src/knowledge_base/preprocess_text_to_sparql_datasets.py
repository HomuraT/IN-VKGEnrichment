import os
import json
from datetime import datetime

def setup_huggingface_mirror():
    """
    设置Hugging Face国内镜像源和缓存目录
    
    Returns:
        None: 无返回值，直接设置环境变量
    """
    # 设置清华大学镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 设置缓存目录
    cache_dir = '/datanfs4/renlin24/cache'
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"已设置Hugging Face镜像源: https://hf-mirror.com")
    print(f"已设置缓存目录: {cache_dir}")

# 在导入datasets之前设置镜像源
setup_huggingface_mirror()

# 现在导入datasets
from datasets import load_dataset, DownloadConfig

def save_knowledge_base(examples: list, filename: str = None) -> str:
    """
    将text到SPARQL的示例保存到知识库文件中
    
    Args:
        examples (list): 包含text和SPARQL对的示例列表
        filename (str, optional): 保存文件名，默认使用时间戳命名
    
    Returns:
        str: 保存文件的完整路径
    """
    # 确保知识库目录存在
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                          "resources", "text_to_sparql_examples")
    os.makedirs(kb_dir, exist_ok=True)
    
    # 生成文件名
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"text_to_sparql_kb_{timestamp}.json"
    
    # 确保文件名以.json结尾
    if not filename.endswith('.json'):
        filename += '.json'
    
    file_path = os.path.join(kb_dir, filename)
    
    # 准备保存的数据
    knowledge_base_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_examples": len(examples),
            "description": "Text to SPARQL knowledge base from PaDaS-Lab/Instruct-to-SPARQL dataset"
        },
        "examples": examples
    }
    
    # 保存到JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base_data, f, indent=2, ensure_ascii=False)
    
    print(f"知识库已保存到: {file_path}")
    print(f"共保存了 {len(examples)} 条text-SPARQL对")
    
    return file_path

def load_instruct_to_sparql_dataset() -> list:
    """
    加载 PaDaS-Lab/Instruct-to-SPARQL 数据集，带镜像回退与离线兜底。
    
    环境变量：
        T2S_SPLIT: 数据切片（默认为 'train'，可设为 'train[:50]' 进行快速验证）
    
    Returns:
        list: 包含 text 和 SPARQL 对的示例列表
    """
    examples = []

    split = os.getenv('T2S_SPLIT', 'train')
    print(f"正在加载PaDaS-Lab/Instruct-to-SPARQL数据集，split={split}...")

    cache_dir = os.environ.get('HF_HOME', '/datanfs4/renlin24/cache')

    def _try_load_with_endpoint(endpoint: str, local_only: bool = False):
        # 切换端点并尝试加载
        if endpoint:
            os.environ['HF_ENDPOINT'] = endpoint
        else:
            os.environ.pop('HF_ENDPOINT', None)

        # 离线模式控制
        if local_only:
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)

        dl_cfg = DownloadConfig(
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=local_only,
            max_retries=1,
        )
        return load_dataset("PaDaS-Lab/Instruct-to-SPARQL", split=split, download_config=dl_cfg)

    ds = None
    last_err = None
    # 1) 先用镜像
    for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
        print(f"尝试端点: {endpoint}")
        ds = _try_load_with_endpoint(endpoint, local_only=False)
        print(f"使用端点成功: {endpoint}")
        break

    # 2) 兜底：离线缓存
    if ds is None:
        print("尝试离线模式（仅使用本地缓存）...")
        ds = _try_load_with_endpoint(None, local_only=True)
        print("离线模式加载成功（使用本地缓存）")

    # 解析样本，字段做兼容处理
    count = 0
    for sample in ds:
        sparql_part = (
            sample.get("sparql_annotated")
            or sample.get("sparql")
            or sample.get("sparql_query")
            or sample.get("query")
        )

        if sparql_part is None:
            continue

        texts = (
            sample.get("instructions")
            or sample.get("questions")
            or ([sample.get("text")] if sample.get("text") else None)
            or ([sample.get("instruction")] if sample.get("instruction") else None)
        )

        if texts is None:
            continue

        if isinstance(texts, str):
            texts = [texts]

        for text_part in texts:
            if text_part is None:
                continue
            examples.append({"text": text_part, "SPARQL": sparql_part})
            count += 1

    print(f"已加载 {len(examples)} 条数据")
    return examples

def load_all_datasets() -> list:
    """
    加载所有可用的text到SPARQL数据集
    
    Returns:
        list: 合并后的所有text和SPARQL对示例列表
    """
    all_examples = []
    
    # 加载PaDaS-Lab/Instruct-to-SPARQL数据集
    instruct_to_sparql_examples = load_instruct_to_sparql_dataset()
    all_examples.extend(instruct_to_sparql_examples)
    
    # TODO: 在这里添加其他数据集
    # 例如:
    # other_dataset_examples = load_other_dataset()
    # all_examples.extend(other_dataset_examples)
    
    print(f"总共合并了 {len(all_examples)} 条text-SPARQL对")
    return all_examples

# 主程序
if __name__ == "__main__":
    # 加载所有数据集
    text_to_sparql_examples = load_all_datasets()
    
    # 保存知识库
    saved_path = save_knowledge_base(text_to_sparql_examples)
    print(f"知识库文件保存完成: {saved_path}")