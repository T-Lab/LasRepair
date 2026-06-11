import openai
import pandas as pd
import numpy as np
import json
import re
import os
import hashlib
from pathlib import Path
from openai import OpenAI
import dotenv

from .paths import DEFAULT_CACHE_DIR, DEFAULT_DATASETS_DIR

dotenv.load_dotenv()
PROMPT_TEMPLATE = """
You are a careful analytical assistant. I will provide you with a dataset containing {n} rows of data with multiple columns.

TASK: 
- For EVERY pair of columns (A, B) where A and B are different, estimate the causal strength between them. 
- You need to consider each pair of columns carefully and give a high strength if you think there is any causal relationship between them.
- You should also give high strength if there is functional dependency between them.

Return exactly ONE JSON object (only JSON, no extra text) where:
- The key is a string in format "column_i->column_j" (where i and j are column indices, i < j)
- The value is an object with two fields:
  * "strength": a float in [0,1] indicating estimated causal strength (0 = no causal relationship, 1 = very strong causal relationship)
  * "confidence": a float in [0,1] indicating how confident you are in this judgement

Important rules:
1) Analyze ALL possible column pairs where i < j (upper triangle of the causal matrix)
2) Use the provided sample data to estimate relationships
3) Be conservative in your estimates
4) Output must be valid JSON only, no explanations

DATA:
Column names: {columns}
Sample data (first {n} rows):
{data}

Example output format:
{{
  "0->1": {{"strength": 0.3, "confidence": 0.6}},
  "0->2": {{"strength": 0.7, "confidence": 0.8}},
  "1->2": {{"strength": 0.1, "confidence": 0.4}}
}}

Please respond now with the JSON only.
""".strip()


def _get_cache_path(dataset_path, cache_dir=None):
    """
    根据数据集路径生成缓存文件路径
    
    Args:
        dataset_path: CSV数据集文件路径
        cache_dir: 缓存目录，如果为None则使用当前目录下的'causal_matrices'子目录
    
    Returns:
        缓存文件的完整路径
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR / "causal_matrices"
    else:
        cache_dir = Path(cache_dir)
    
    # 确保缓存目录存在
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 基于数据集路径生成唯一文件名
    dataset_path_str = str(Path(dataset_path).resolve())
    # 使用路径的hash值来确保唯一性
    path_hash = hashlib.md5(dataset_path_str.encode()).hexdigest()[:8]
    dataset_name = Path(dataset_path).stem
    cache_filename = f"{dataset_name}_{path_hash}_causal_matrix.npy"
    
    return cache_dir / cache_filename


def generate_causal_matrix(
    dataset_path,
    cache_dir=None,
    n_rows=20,
    random_state=114,
    force_regenerate=False,
    verbose=True
):
    """
    生成数据集的因果矩阵
    
    该函数会先检查缓存目录中是否存在对应的npy文件，如果存在则直接加载，
    否则调用LLM API进行分析，生成因果矩阵并保存到缓存目录。
    
    Args:
        dataset_path: CSV数据集文件路径
        cache_dir: 缓存目录，如果为None则使用当前目录下的'causal_matrices'子目录
        n_rows: 用于分析的采样行数，默认20
        random_state: 随机种子，默认114
        force_regenerate: 是否强制重新生成，忽略缓存，默认False
        verbose: 是否打印详细信息，默认True
    
    Returns:
        numpy.ndarray: 因果矩阵，形状为 (n_columns, n_columns)
    """
    # 获取缓存文件路径
    cache_path = _get_cache_path(dataset_path, cache_dir)
    
    # 检查缓存是否存在
    if not force_regenerate and cache_path.exists():
        if verbose:
            print(f"从缓存加载因果矩阵: {cache_path}")
        return np.load(cache_path)
    
    # 读取数据集
    if verbose:
        print(f"读取数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 准备采样数据
    sample_n_rows = min(n_rows, len(df))
    sample_data = df.sample(sample_n_rows, random_state=random_state)
    
    # 构建prompt
    input_text = PROMPT_TEMPLATE.format(
        columns=list(df.columns),
        n=sample_n_rows,
        data=sample_data
    )
    
    if verbose:
        print(f"准备调用API，分析 {len(df.columns)} 列之间的所有关系...")
        print(f"总共需要分析 {len(df.columns) * (len(df.columns) - 1) // 2} 个列对")
    
    # 调用API
    client = OpenAI(
        base_url=os.getenv("LIGHTNING_BASE_URL", "https://lightning.ai/api/v1/"),
        api_key=os.environ.get("LIGHTNING_API_KEY"),
    )
    
    response = client.chat.completions.create(
        model="openai/gpt-5",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": input_text}]
            },
        ],
    )
    
    # 提取响应
    response_text = response.choices[0].message.content
    if verbose:
        print("\n=== API响应 ===")
        print(response_text)
        print("================\n")
    
    # 初始化因果矩阵
    causal_matrix = np.zeros((len(df.columns), len(df.columns)))
    
    # 解析JSON响应
    try:
        # 尝试找到JSON部分
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # 遍历所有返回的列对关系
            for key, value in result.items():
                # 解析 "i->j" 格式
                match = re.match(r'(\d+)->(\d+)', key)
                if match:
                    i = int(match.group(1))
                    j = int(match.group(2))
                    strength = value.get('strength', 0.0)
                    confidence = value.get('confidence', 0.0)
                    causal_matrix[i, j] = strength
                    if verbose:
                        print(f"列 {i} ({df.columns[i]}) -> 列 {j} ({df.columns[j]}): strength={strength:.3f}, confidence={confidence:.3f}")
                else:
                    if verbose:
                        print(f"警告: 无法解析键 '{key}'")
            
            if verbose:
                print(f"\n成功解析 {len(result)} 个列对关系")
        else:
            if verbose:
                print("错误: 在响应中未找到JSON")
            raise ValueError("API响应中未找到有效的JSON数据")
            
    except json.JSONDecodeError as e:
        if verbose:
            print(f"JSON解析错误: {e}")
            print("响应文本:", response_text)
        raise ValueError(f"无法解析API响应为JSON: {e}")
    
    # 保存到缓存
    np.save(cache_path, causal_matrix)
    if verbose:
        print(f"\n因果矩阵生成完成!")
        print(f"矩阵已保存到: {cache_path}")
    
    return causal_matrix


# 为了向后兼容，保留原有的执行逻辑（如果直接运行此文件）
if __name__ == "__main__":
    clean_csv = DEFAULT_DATASETS_DIR / "beers" / "clean.csv"
    causal_matrix = generate_causal_matrix(clean_csv)
    print("\n因果矩阵:")
    print(causal_matrix)
