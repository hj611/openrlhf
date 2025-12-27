import torch
import json
import os
import requests
from datetime import datetime
from typing import Dict, List, Tuple

# 配置路径
LOG_PATH = os.environ.get("REWARD_LOG_PATH", "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/code/OpenRLHF/reward.log")
URL_MAPPING_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/code/OpenRLHF/data/result.json"

# 全局缓存 URL 映射
_url_mapping_cache = None

# 代理配置
PROXIES = {
    'http': 'http://10.229.18.23:3128',
    'https': 'http://10.229.18.23:3128'
}

# 接受的状态（大小写不敏感）
ACCEPTED_STATUSES = {'accept', 'accepted', 'ac'}


def load_url_mapping(force_reload: bool = False) -> Dict[str, str]:
    """
    加载 URL 映射配置
    
    Args:
        force_reload: 是否强制重新加载
        
    Returns:
        Dict[str, str]: problem_id -> base_url 的映射
    """
    global _url_mapping_cache
    
    if _url_mapping_cache is None or force_reload:
        try:
            with open(URL_MAPPING_PATH, 'r', encoding='utf-8') as f:
                _url_mapping_cache = json.load(f)
            print(f"✓ 成功加载 URL 映射，共 {len(_url_mapping_cache)} 个问题")
        except FileNotFoundError:
            print(f"⚠️  警告: URL 映射文件不存在 - {URL_MAPPING_PATH}")
            _url_mapping_cache = {}
        except json.JSONDecodeError as e:
            print(f"❌ 错误: URL 映射文件格式错误 - {e}")
            _url_mapping_cache = {}
    
    return _url_mapping_cache


def extract_problem_id(answer: str) -> str:
    """
    从 answer 中提取 problem_id
    
    Args:
        answer: 答案字符串（可能包含 problem_id）
        
    Returns:
        str: 提取的 problem_id，如果提取失败返回空字符串
    """
    # 方法1: 如果 answer 本身就是 problem_id
    if isinstance(answer, str) and '_' in answer:
        return answer.strip()
    
    # 方法2: 如果 answer 是 JSON 字符串
    try:
        answer_data = json.loads(answer)
        if isinstance(answer_data, dict):
            return answer_data.get('problem_id', '')
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 方法3: 使用正则表达式提取
    import re
    match = re.search(r'(\d+_[A-Z])', answer)
    if match:
        return match.group(1)
    
    return ""


def calculate_reward(api_result: Dict) -> Tuple[float, str]:
    """
    根据 API 返回结果计算 reward
    
    Args:
        api_result: API 返回的结果字典
        
    Returns:
        Tuple[float, str]: (reward 值, 状态说明)
    """
    status = api_result.get('status', '').lower().strip()
    
    # 检查是否为接受状态
    if status in ACCEPTED_STATUSES:
        return 1.0, f"✅ Accepted ({status})"
    else:
        # 其他所有状态都返回 0
        original_status = api_result.get('status', 'Unknown')
        return 0.0, f"❌ {original_status}"


def get_reward_from_api(base_url: str, problem_id: str, code: str,  
                        timeout: int = 600) -> Tuple[float, Dict]:
    """
    通过 API 获取 reward
    
    Args:
        base_url: API 基础 URL
        problem_id: 问题 ID
        code: 提交的代码
        timeout: 请求超时时间（秒）
        
    Returns:
        Tuple[float, Dict]: (reward 值, 额外信息)
    """
    try:
        # 构建完整 URL
        url = f"{base_url}/api/submit/sync"
        
        # 准备请求数据
        data = {
            "problem_id": problem_id,
            "code": code,
            "language": 'c++17',
        }
        
        print(f"🔄 发送请求到: {url}")
        print(f"   Problem ID: {problem_id}")
        print(f"   Code length: {len(code)} chars")
        
        # 发送 POST 请求
        api_response = requests.post(
            url, 
            data=data,
            timeout=timeout,
            proxies=PROXIES,
        )
        
        # 检查响应状态
        api_response.raise_for_status()
        
        # 解析响应
        result = api_response.json()
        
        print(f"📥 API 响应:")
        print(f"   Status: {result.get('status', 'Unknown')}")
        print(f"   Score: {result.get('score', 0)}")
        print(f"   Time: {result.get('time_used', 0)}ms")
        print(f"   Memory: {result.get('memory_used', 0)}KB")
        
        # 计算 reward
        reward, status_msg = calculate_reward(result)
        
        print(f"   Reward: {reward} - {status_msg}")
        
        # 如果有错误信息，打印出来
        if result.get('message'):
            print(f"   Message: {result['message'][:200]}...")
        
        # 额外信息
        extra_info = {
            "status": "success",
            "judge_status": result.get('status', 'Unknown'),
            "score": result.get('score', 0),
            "time_used": result.get('time_used', 0),
            "memory_used": result.get('memory_used', 0),
            "message": result.get('message', ''),
            "failed_case": result.get('failed_case', 0),
            "problem_id": problem_id,
            "submission_id": result.get('id', 0),
            "api_response": result
        }
        
        return reward, extra_info
        
    except requests.exceptions.Timeout:
        print(f"⚠️  API 请求超时: {url} - {problem_id}")
        return 0.0, {
            "status": "timeout", 
            "problem_id": problem_id,
            "judge_status": "Timeout"
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API 请求失败: {e}")
        return 0.0, {
            "status": "error", 
            "error": str(e), 
            "problem_id": problem_id,
            "judge_status": "Request Error"
        }
        
    except (ValueError, KeyError) as e:
        print(f"❌ 解析响应失败: {e}")
        return 0.0, {
            "status": "parse_error", 
            "error": str(e), 
            "problem_id": problem_id,
            "judge_status": "Parse Error"
        }


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor or List[str]): Complete text sequences containing prompts and responses (代码)
        prompts (torch.Tensor or List[str]): Input prompt sequences
        labels (torch.Tensor or List[str]): Ground truth answer sequences (problem_ids)
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """
    
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 加载 URL 映射
    url_mapping = load_url_mapping()
    
    # 存储所有 reward
    rewards_list = []
    extra_logs_list = []
    
    # 统计信息
    status_counter = {}
    
    # 打开日志文件
    print(f"📝 日志路径: {LOG_PATH}")
    with open(LOG_PATH, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Batch Evaluation - {current_time}\n")
        f.write(f"{'='*80}\n\n")
        
        # 遍历每个样本
        for idx, (query, prompt, answer) in enumerate(zip(queries, prompts, labels)):
            f.write(f"\n{'─'*80}\n")
            f.write(f"Sample {idx + 1}/{len(queries)}\n")
            f.write(f"{'─'*80}\n")
            
            # 解析 prompt（如果是 JSON）
            prompt_data = None
            try:
                if isinstance(prompt, str):
                    prompt_data = json.loads(prompt)
                    f.write(f"✓ Prompt 解析成功\n")
                    f.write(f"Prompt Data: {json.dumps(prompt_data, indent=2, ensure_ascii=False)}\n\n")
            except json.JSONDecodeError as e:
                f.write(f"⚠️  Prompt JSON 解码错误: {e}\n")
                f.write(f"Raw Prompt: {prompt[:200]}...\n\n")
            
            # 提取 problem_id
            problem_id = extract_problem_id(answer)
            
            if not problem_id:
                f.write(f"❌ 无法提取 problem_id from answer: {answer}\n")
                rewards_list.append(0.0)
                extra_logs_list.append({
                    "status": "no_problem_id", 
                    "answer": str(answer),
                    "judge_status": "Invalid Problem ID"
                })
                status_counter["Invalid Problem ID"] = status_counter.get("Invalid Problem ID", 0) + 1
                continue
            
            f.write(f"Problem ID: {problem_id}\n")
            
            # 查找对应的 base_url
            base_url = url_mapping.get(problem_id)
            
            if not base_url:
                f.write(f"⚠️  未找到 problem_id 对应的 URL: {problem_id}\n")
                f.write(f"可用的 problem_ids: {list(url_mapping.keys())[:10]}...\n")
                rewards_list.append(0.0)
                extra_logs_list.append({
                    "status": "url_not_found", 
                    "problem_id": problem_id,
                    "judge_status": "URL Not Found"
                })
                status_counter["URL Not Found"] = status_counter.get("URL Not Found", 0) + 1
                continue
            
            f.write(f"Base URL: {base_url}\n")
            
            # query 就是代码
            code = str(query).strip()
            
            f.write(f"\n===the gen Code: {code}\n")
            # f.write(f"Code length: {len(code)} chars\n")
            f.write(f"Answer (Problem ID): {answer}\n\n")
            
            # 调用 API 获取 reward
            f.write(f"🔄 正在请求 API...\n")
            reward, extra_info = get_reward_from_api(
                base_url=base_url,
                problem_id=problem_id,
                code=code
            )
            
            f.write(f"✓ Reward: {reward}\n")
            f.write(f"Judge Status: {extra_info.get('judge_status', 'Unknown')}\n")
            f.write(f"Score: {extra_info.get('score', 0)}\n")
            f.write(f"Time Used: {extra_info.get('time_used', 0)}ms\n")
            f.write(f"Memory Used: {extra_info.get('memory_used', 0)}KB\n")
            
            if extra_info.get('message'):
                f.write(f"Message: {extra_info['message'][:500]}...\n")
            
            f.write(f"\nExtra Info: {json.dumps(extra_info, indent=2, ensure_ascii=False)}\n")
            
            rewards_list.append(reward)
            extra_logs_list.append(extra_info)
            
            # 统计状态
            judge_status = extra_info.get('judge_status', 'Unknown')
            status_counter[judge_status] = status_counter.get(judge_status, 0) + 1
    
    # 转换为 tensor
    rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
    
    # scores 就是 rewards（因为已经是 0 或 1）
    scores_tensor = rewards_tensor.clone()
    
    # 计算统计信息
    total_samples = len(rewards_list)
    accepted_count = sum(1 for r in rewards_list if r > 0)
    avg_reward = sum(rewards_list) / total_samples if total_samples > 0 else 0
    success_rate = (accepted_count / total_samples * 100) if total_samples > 0 else 0
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"📊 Batch Evaluation Summary - {current_time}")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples}")
    print(f"Accepted: {accepted_count} ({success_rate:.2f}%)")
    print(f"Failed: {total_samples - accepted_count} ({100 - success_rate:.2f}%)")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"\n状态分布:")
    for status, count in sorted(status_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {status}: {count} ({percentage:.2f}%)")
    print(f"{'='*80}\n")
    
    return {
        "rewards": rewards_tensor,  # Rewards for advantage calculation (0 or 1)
        "scores": scores_tensor,    # Scores for dynamic filtering (0 or 1)
        "extra_logs": {
            "reward_details": extra_logs_list,
            "avg_reward": avg_reward,
            "max_reward": 1.0,
            "min_reward": 0.0,
            "accepted_count": accepted_count,
            "total_count": total_samples,
            "success_rate": success_rate,
            "status_distribution": status_counter
        }
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("🧪 开始测试 reward_func\n")
    
    # 测试数据
    test_queries = [
        # 测试1: 有编译错误的代码（应该返回 0）
        "#include <bits/stdc++.h>\n#pragma comment(linker, \"/STACK:2000000\")\n#pragma comment(linker, \"/HEAP:2000000\")\nusing namespace std;\nint32_t main() {\n  ios_base::sync_with_stdio(false);\n  cin.tie(NULL);\n  cout.tie(NULL);\n  int n;\n  cin >> n;\n  long long m[n][n], a[n];\n  for (int i = 0; i < n; i++) {\n    for (int j = 0; j < n; j++) {\n      cin >> m[i][j];\n    }\n  }\n  long long x = sqrt((m[0][1] * m[0][2]) \\/ m[1][2]);\n  cout << x << \" \";\n  for (int i = 1; i < n; i++) {\n    cout << m[0][i] \\/ x << \" \";\n  }\n  return 0;\n}\n",
        
        # 测试2: 正确的代码（应该返回 1，如果通过所有测试）
        "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  int n;\n  cin >> n;\n  long long m[n][n];\n  for (int i = 0; i < n; i++) {\n    for (int j = 0; j < n; j++) {\n      cin >> m[i][j];\n    }\n  }\n  long long x = sqrt((m[0][1] * m[0][2]) / m[1][2]);\n  cout << x << \" \";\n  for (int i = 1; i < n; i++) {\n    cout << m[0][i] / x << \" \";\n  }\n  return 0;\n}\n",
    ]
    
    test_prompts = [
        '{"problem_id": "1220_B", "description": "..."}',
        '{"problem_id": "1220_B", "description": "..."}',
    ]
    
    test_labels = [
        "1220_B",
        "1220_B",
    ]
    
    # 调用 reward 函数
    result = reward_func(test_queries, test_prompts, test_labels)
    
    print("\n" + "="*80)
    print("🎯 测试结果:")
    print("="*80)
    print(f"Rewards: {result['rewards']}")
    print(f"Scores: {result['scores']}")
    print(f"\nExtra logs:")
    print(f"  Average reward: {result['extra_logs']['avg_reward']:.4f}")
    print(f"  Accepted: {result['extra_logs']['accepted_count']}/{result['extra_logs']['total_count']}")
    print(f"  Success rate: {result['extra_logs']['success_rate']:.2f}%")
    print(f"\n  Status distribution:")
    for status, count in result['extra_logs']['status_distribution'].items():
        print(f"    {status}: {count}")
    print("="*80)