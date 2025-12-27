import torch
import json
import os
import requests
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 配置
# ============================================================================
LOG_PATH = os.environ.get(
    "REWARD_LOG_PATH", 
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/code/OpenRLHF/reward.log"
)
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

problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"

# 🔧 添加：每个服务器的请求间隔（秒）
REQUEST_INTERVAL_PER_SERVER = 1.0  # 同一服务器的请求间隔 1 秒
_last_request_time_per_server = {}  # 记录每个服务器的最后请求时间


# ============================================================================
# 辅助函数
# ============================================================================

def load_url_mapping(force_reload: bool = False) -> Dict[str, str]:
    """加载 URL 映射配置"""
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


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end():]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def extract_code_from_output(text: str) -> Optional[str]:
    """从模型输出中提取代码块"""
    
    # 方法1: 提取 Markdown 代码块（带语言标记）
    pattern1 = r'```(?:cpp|c\+\+|c|python|java|javascript|go|rust)\s*\n(.*?)```'
    matches = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        code = matches[-1].strip()
        print(f"✓ 提取到代码块（方法1）: {len(code)} 字符")
        return code
    
    # 方法2: 提取无语言标记的代码块
    pattern2 = r'```\s*\n(.*?)```'
    matches = re.findall(pattern2, text, re.DOTALL)
    
    if matches:
        code = matches[-1].strip()
        print(f"✓ 提取到代码块（方法2）: {len(code)} 字符")
        return code
    
    # 方法3: 查找 #include 开头的 C++ 代码
    pattern3 = r'(#include\s+<[^>]+>.*?)(?:\n\n[A-Z]|\Z)'
    matches = re.findall(pattern3, text, re.DOTALL)
    
    if matches:
        code = matches[-1].strip()
        print(f"✓ 提取到代码块（方法3）: {len(code)} 字符")
        return code
    
    print("⚠️  未能提取到代码块")
    return None


def extract_problem_id(answer: str) -> str:
    """从 answer 中提取 problem_id"""
    if isinstance(answer, str) and '_' in answer:
        return answer.strip()
    
    try:
        answer_data = json.loads(answer)
        if isinstance(answer_data, dict):
            return answer_data.get('problem_id', '')
    except (json.JSONDecodeError, TypeError):
        pass
    
    match = re.search(r'(\d+_[A-Z])', answer)
    if match:
        return match.group(1)
    
    return ""


def calculate_reward(api_result: Dict) -> Tuple[float, str]:
    """根据 API 返回结果计算 reward"""
    status = api_result.get('status', '').lower().strip()
    
    if status in ACCEPTED_STATUSES:
        return 1.0, f"✅ Accepted ({status})"
    else:
        original_status = api_result.get('status', 'Unknown')
        return 0.0, f"❌ {original_status}"

import random  # 导入 random 模块

RANDOM_WAIT_MIN = 0.0  # 最小等待时间
RANDOM_WAIT_MAX = 1.0  # 最大等待时间

def random_wait():
    """随机等待 0-1 秒"""
    wait_time = random.uniform(RANDOM_WAIT_MIN, RANDOM_WAIT_MAX)
    print(f"🎲 随机等待 {wait_time:.3f} 秒...")
    time.sleep(wait_time)


def get_reward_from_api(
    base_url: str, 
    problem_id: str, 
    code: str,  
    timeout: int = 600
) -> Tuple[float, Dict]:
    """
    通过 API 获取 reward（串行，带间隔控制）
    """
    try:
        
        url = f"{base_url}/api/submit/sync"
        
        data = {
            "problem_id": problem_id,
            "code": code,
            "language": 'c++17',
        }
        
        print(f"🔄 发送请求到: {url}")
        print(f"   Problem ID: {problem_id}")
        print(f"   Code length: {len(code)} chars")
        print(f"   时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 🔧 发送请求并等待响应
        request_start_time = time.time()
        
        api_response = requests.post(
            url, 
            data=data,
            timeout=timeout,
            proxies=PROXIES,
        )
        
        request_duration = time.time() - request_start_time
        
        api_response.raise_for_status()
        result = api_response.json()
        
        print(f"📥 API 响应 (耗时 {request_duration:.2f}s):")
        print(f"   Status: {result.get('status', 'Unknown')}")
        print(f"   Score: {result.get('score', 0)}")
        
        reward, status_msg = calculate_reward(result)
        print(f"   Reward: {reward} - {status_msg}")
        
        # 返回标量值
        extra_info = {
            "status": "success",
            "judge_status": result.get('status', 'Unknown'),
            "score": float(result.get('score', 0)),
            "time_used": float(result.get('time_used', 0)),
            "memory_used": float(result.get('memory_used', 0)),
            "failed_case": float(result.get('failed_case', 0)),
            "problem_id": problem_id,
            "submission_id": float(result.get('id', 0)),
            "request_duration": request_duration,  # 🔧 记录请求耗时
        }
        
        return reward, extra_info
        
    except requests.exceptions.Timeout:
        print(f"⚠️  API 请求超时 (>{timeout}s)")
        return 0.0, {
            "status": "timeout", 
            "judge_status": "Timeout",
            "score": 0.0,
            "time_used": 0.0,
            "memory_used": 0.0,
            "failed_case": 0.0,
            "problem_id": problem_id,
            "submission_id": 0.0,
            "request_duration": timeout,
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API 请求失败: {e}")
        return 0.0, {
            "status": "error", 
            "judge_status": "Request Error",
            "score": 0.0,
            "time_used": 0.0,
            "memory_used": 0.0,
            "failed_case": 0.0,
            "problem_id": problem_id,
            "submission_id": 0.0,
            "request_duration": 0.0,
        }
        
    except (ValueError, KeyError) as e:
        print(f"❌ 解析响应失败: {e}")
        return 0.0, {
            "status": "parse_error", 
            "judge_status": "Parse Error",
            "score": 0.0,
            "time_used": 0.0,
            "memory_used": 0.0,
            "failed_case": 0.0,
            "problem_id": problem_id,
            "submission_id": 0.0,
            "request_duration": 0.0,
        }


# ============================================================================
# 主函数
# ============================================================================

def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    真正并行处理：将样本按服务器分组，每台服务器独立串行处理自己的队列。

    Args:
        queries: 模型的完整输出（包含代码）
        prompts: 输入提示
        labels: 标准答案（problem_id）

    Returns:
        dict: {
            "rewards": Tensor[batch_size],
            "scores": Tensor[batch_size],
            "extra_logs": Dict[str, Tensor[batch_size]]
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    url_mapping = load_url_mapping()

    batch_size = len(queries)

    # 收集指标 - 使用字典按索引存储，确保顺序
    results = {}
    results_lock = threading.Lock()

    # 统计信息
    status_counter = {}
    code_extraction_stats = {"success": 0, "failed": 0}
    stats_lock = threading.Lock()

    # 从url_mapping中提取所有唯一的base_url
    unique_base_urls = list(set(url_mapping.values()))

    # 按服务器分组样本
    server_samples = {url: [] for url in unique_base_urls}

    for idx, (query, prompt, answer) in enumerate(zip(queries, prompts, labels)):
        problem_id = extract_problem_id(answer)
        if problem_id:
            base_url = url_mapping.get(problem_id)
            if base_url and base_url in server_samples:
                server_samples[base_url].append((idx, query, prompt, answer))

    print(f"📝 处理 {batch_size} 个样本（真并行模式，{len(unique_base_urls)} 台服务器）")
    print(f"📝 服务器列表: {unique_base_urls}")
    for url, samples in server_samples.items():
        print(f"   {url}: {len(samples)} 个样本")
    print(f"📝 日志路径: {LOG_PATH}")
    print(f"📝 每台服务器请求间隔: {REQUEST_INTERVAL_PER_SERVER} 秒")

    batch_start_time = time.time()

    def process_server_queue(base_url, samples):
        """处理单台服务器的所有样本（串行）"""
        log_entries = []

        log_entries.append(f"\n{'='*80}\n")
        log_entries.append(f"Server: {base_url}\n")
        log_entries.append(f"Samples: {len(samples)}\n")
        log_entries.append(f"{'='*80}\n")

        for idx, query, prompt, answer in samples:
            sample_start_time = time.time()

            log_entries.append(f"\n{'─'*80}\n")
            log_entries.append(f"Sample {idx + 1}/{batch_size} - {datetime.now().strftime('%H:%M:%S')}\n")
            log_entries.append(f"Server: {base_url}\n")
            log_entries.append(f"{'─'*80}\n")

            response = get_response_from_query(query)

            if not response:
                print(f"⚠️  [{base_url}] 样本 {idx}: 未能提取到 assistant 回复")
                result = {
                    "reward": 0.0,
                    "score": 0.0,
                    "time_used": 0.0,
                    "memory_used": 0.0,
                    "failed_case": 0.0,
                    "submission_id": 0.0,
                    "request_duration": 0.0,
                    "judge_status": "No Response"
                }
                with results_lock:
                    results[idx] = result
                with stats_lock:
                    status_counter["No Response"] = status_counter.get("No Response", 0) + 1
                continue

            # 提取 problem_id
            problem_id = extract_problem_id(answer)

            if not problem_id:
                log_entries.append(f"❌ 无法提取 problem_id from answer: {answer}\n")
                result = {
                    "reward": 0.0,
                    "score": 0.0,
                    "time_used": 0.0,
                    "memory_used": 0.0,
                    "failed_case": 0.0,
                    "submission_id": 0.0,
                    "request_duration": 0.0,
                    "judge_status": "Invalid Problem ID"
                }
                with results_lock:
                    results[idx] = result
                with stats_lock:
                    status_counter["Invalid Problem ID"] = status_counter.get("Invalid Problem ID", 0) + 1
                continue

            log_entries.append(f"Problem ID: {problem_id}\n")

            # 从输出中提取代码
            full_output = str(response).strip()
            code = extract_code_from_output(full_output)

            if code is None:
                log_entries.append(f"❌ 未能从输出中提取代码块\n")
                code = full_output
                with stats_lock:
                    code_extraction_stats["failed"] += 1
            else:
                log_entries.append(f"✓ 成功提取代码块\n")
                with stats_lock:
                    code_extraction_stats["success"] += 1

            log_entries.append(f"\n===full_output: \n{full_output[:500]}...\n\n")
            log_entries.append(f"Code length: {len(code)} chars\n")
            log_entries.append(f"\n===Extracted Code:\n{code[:500]}...\n\n")

            # 限流：等待间隔
            time.sleep(REQUEST_INTERVAL_PER_SERVER)

            log_entries.append(f"🔄 正在请求 API...\n")

            reward, extra_info = get_reward_from_api(
                base_url=base_url,
                problem_id=problem_id,
                code=code
            )

            sample_duration = time.time() - sample_start_time

            log_entries.append(f"✓ Reward: {reward}\n")
            log_entries.append(f"Judge Status: {extra_info.get('judge_status', 'Unknown')}\n")
            log_entries.append(f"Score: {extra_info.get('score', 0)}\n")
            log_entries.append(f"Time Used: {extra_info.get('time_used', 0)} ms\n")
            log_entries.append(f"Memory Used: {extra_info.get('memory_used', 0)} KB\n")
            log_entries.append(f"Request Duration: {extra_info.get('request_duration', 0):.2f}s\n")
            log_entries.append(f"Sample Total Duration: {sample_duration:.2f}s\n")

            # 保存结果
            result = {
                "reward": reward,
                "score": extra_info['score'],
                "time_used": extra_info['time_used'],
                "memory_used": extra_info['memory_used'],
                "failed_case": extra_info['failed_case'],
                "submission_id": extra_info['submission_id'],
                "request_duration": extra_info.get('request_duration', 0),
                "judge_status": extra_info.get('judge_status', 'Unknown')
            }

            with results_lock:
                results[idx] = result

            with stats_lock:
                judge_status = extra_info.get('judge_status', 'Unknown')
                status_counter[judge_status] = status_counter.get(judge_status, 0) + 1

            print(f"✓ [{base_url}] 完成 {idx + 1}/{batch_size} (耗时 {sample_duration:.2f}s)")

        return log_entries

    # 并行处理：每台服务器一个线程
    with open(LOG_PATH, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Batch Evaluation (True Parallel) - {current_time}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Servers: {len(unique_base_urls)}\n")
        f.write(f"Request Interval per Server: {REQUEST_INTERVAL_PER_SERVER}s\n")
        f.write(f"{'='*80}\n\n")

        # 为每台服务器启动一个线程
        with ThreadPoolExecutor(max_workers=len(unique_base_urls)) as executor:
            futures = {
                executor.submit(process_server_queue, base_url, samples): base_url
                for base_url, samples in server_samples.items()
                if samples  # 只处理有样本的服务器
            }

            # 收集日志
            for future in as_completed(futures):
                base_url = futures[future]
                try:
                    log_entries = future.result()
                    f.writelines(log_entries)
                    f.flush()
                    print(f"✓ 服务器 {base_url} 完成所有样本")
                except Exception as e:
                    print(f"❌ 服务器 {base_url} 处理失败: {e}")
                    f.write(f"❌ Server {base_url} failed: {e}\n")

    batch_duration = time.time() - batch_start_time

    # 按索引顺序提取结果
    rewards_list = [results.get(i, {"reward": 0.0})["reward"] for i in range(batch_size)]
    scores_list = [results.get(i, {"score": 0.0})["score"] for i in range(batch_size)]
    time_used_list = [results.get(i, {"time_used": 0.0})["time_used"] for i in range(batch_size)]
    memory_used_list = [results.get(i, {"memory_used": 0.0})["memory_used"] for i in range(batch_size)]
    failed_case_list = [results.get(i, {"failed_case": 0.0})["failed_case"] for i in range(batch_size)]
    submission_id_list = [results.get(i, {"submission_id": 0.0})["submission_id"] for i in range(batch_size)]
    request_duration_list = [results.get(i, {"request_duration": 0.0})["request_duration"] for i in range(batch_size)]

    # 转换为 tensor
    rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
    scores_tensor = torch.tensor(scores_list, dtype=torch.float32)

    # 计算统计信息
    accepted_count = sum(1 for r in rewards_list if r > 0)
    avg_reward = sum(rewards_list) / batch_size if batch_size > 0 else 0
    success_rate = (accepted_count / batch_size * 100) if batch_size > 0 else 0
    avg_request_duration = sum(request_duration_list) / len(request_duration_list) if request_duration_list else 0

    # 计算理论串行时间和实际加速比
    total_request_time = sum(request_duration_list)
    theoretical_speedup = total_request_time / batch_duration if batch_duration > 0 else 1

    # 统计每台服务器的使用情况
    server_usage = {}
    for base_url, samples in server_samples.items():
        server_usage[base_url] = len(samples)

    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"📊 Batch Evaluation Summary - {current_time}")
    print(f"{'='*80}")
    print(f"Total samples: {batch_size}")
    print(f"Batch duration: {batch_duration:.2f}s")
    print(f"Total request time: {total_request_time:.2f}s")
    print(f"Average request duration: {avg_request_duration:.2f}s")
    print(f"Actual speedup: {theoretical_speedup:.2f}x")
    print(f"Accepted: {accepted_count} ({success_rate:.2f}%)")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"\n代码提取统计:")
    print(f"  成功: {code_extraction_stats['success']}")
    print(f"  失败: {code_extraction_stats['failed']}")
    print(f"\n服务器使用分布:")
    for server, count in sorted(server_usage.items()):
        percentage = (count / batch_size * 100) if batch_size > 0 else 0
        print(f"  {server}: {count} ({percentage:.2f}%)")
    print(f"\n状态分布:")
    for status, count in sorted(status_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / batch_size * 100) if batch_size > 0 else 0
        print(f"  {status}: {count} ({percentage:.2f}%)")
    print(f"{'='*80}\n")

    return {
        "rewards": rewards_tensor,
        "scores": scores_tensor,
        "extra_logs": {
            "time_used": torch.tensor(time_used_list, dtype=torch.float32),
            "memory_used": torch.tensor(memory_used_list, dtype=torch.float32),
            "failed_case": torch.tensor(failed_case_list, dtype=torch.float32),
            "submission_id": torch.tensor(submission_id_list, dtype=torch.float32),
            "request_duration": torch.tensor(request_duration_list, dtype=torch.float32),
        }
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("🧪 开始测试 reward_func (串行模式)\n")
    
    test_query1 = """<|im_start|>assistant
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    cout << n << endl;
    return 0;
}
```<|im_end|>"""
    
    test_queries = [test_query1, test_query1]
    test_prompts = ['{"problem_id": "1220_B"}', '{"problem_id": "1220_B"}']
    test_labels = ["1220_B", "1220_B"]
    
    print(f"测试样本数: {len(test_queries)}\n")
    
    result = reward_func(test_queries, test_prompts, test_labels)
    
    print("\n🎯 测试结果:")
    print(f"Rewards: {result['rewards']}")
    print(f"Request durations: {result['extra_logs']['request_duration']}")