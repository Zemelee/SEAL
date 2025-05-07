# 评估数学推理任务中模型 model_generation 与标准答案（ground truth）是否一致
import os
import json
from tqdm import tqdm, trange
from eval_math_rule.evaluation.grader import math_equal
from eval_math_rule.evaluation.parser import extract_answer, parse_ground_truth, strip_string
from collections import Counter

import multiprocessing
import queue

def math_equal_with_timeout(pred, gt_ans, timeout):
    # 在子进程中执行 math_equal 函数，并将结果放入队列中
    def target(result_queue):
        try:
            result_queue.put(math_equal(pred, gt_ans))
        except Exception as e:
            result_queue.put(e)
    # 创建消息队列，用于存储结果 子进程存/主进程取
    result_queue = multiprocessing.Queue()
    # 创建子进程用于运行target，result_queue作为参数传入
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()
    process.join(timeout) # 主进程等待子进程完成

    if process.is_alive():
        print(f"Timeout occurred for prediction: {pred}")
        process.terminate() # 终止子进程
        process.join() # 等待子进程完全结束
        return False

   
    try:
        result = result_queue.get_nowait() # 非阻塞地从队列里获取结果
    except queue.Empty:
        print("Result queue timed out")
        return False

    if isinstance(result, Exception):
        print(f"Error occurred: {result}")
        return False

    return result


def parallel_math_equal(all_pred, gt_ans, timeout=20):
    # 本质仍然是串行执行
    results = []
    for pred in all_pred:
        results.append(math_equal_with_timeout(pred, gt_ans, timeout))
    return results



def main(res_path, save=False, k=None, output_dir=None):
    # args = parse_args()
    with open(res_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    for example in tqdm(data):
        # gt_cot, gt = parse_ground_truth(example, data_name="omni-math")
        # 兼容性处理：如果没有 model_generation 字段，就用 model_output
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None: # 只取前 k 个预测结果
            example["model_generation"] = example["model_generation"][:k]
        
        gt_cot = example["answer"] # 解决过程
        gt_ans = extract_answer(gt_cot, data_name="omni-math")
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        all_pred = [extract_answer(p, data_name="omni-math") for p in example["model_generation"]]
        all_pred = [strip_string(p, skip_unit=False) for p in all_pred] # 
        # all_eval = [math_equal(p, gt_ans) for p in all_pred]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5) # [T F T...]
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        # 多数投票选择最佳预测
        counter = Counter(effective_pred) # 统计 effective_pred 中每个预测的出现次数
        pred = counter.most_common(1)[0][0] # 选择出现次数最多的预测作为最终预测结果
        index = all_pred.index(pred) # 找到该预测在 all_pred 中的index
        eval = all_eval[index] # 获取对应的评估结果，表示该预测是否正确
        example["all_pred"] = all_pred # 所有预测答案
        example["all_eval"] = all_eval # 所有预测的评估结果 [bool]
        example["mv_pred"] = pred # 最佳预测结果 一个答案str
        example["mv_eval"] = eval # 最佳预测的评估 一个bool
        example["mv_index"] = index # 最佳预测在 all_pred 中的索引(0)

    acc = sum([example["mv_eval"] for example in data]) / len(data)
    print(f"Accuracy: {acc:.3f}")
    
    if save:
        # prompt answer solution model_generation all_pred all_eval mv_pred mv_eval mv_index
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        
        metric_file= os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": acc}, f)



#  model_generation = [
#     "The answer is $\boxed{42}$.",
#     "I think the answer is 42.",
#     "The result should be $\boxed{6}$."
# ]
# │
# │ extract_answer()
# ▼
# all_pred = ["42", "42", "6"]
# │
# │ strip_string()
# ▼
# all_pred = ["42", "42", "6"]
# │
# │ parallel_math_equal(..., gt_ans="42")
# ▼
# all_eval = [True, True, False]
# │
# │ effective_pred: 只保留包含 boxed 的预测（若没有就保留全部）
# │ └──> ["42", "42", "6"] （假设前两个有 boxed）
# │
# │ Counter(effective_pred).most_common(1)
# │ └──> [("42", 2)]
# ▼
# pred = "42"
# index = all_pred.index("42") → 0
# eval = all_eval[0] → True
# │
# │ 最终添加字段：
# │ example["all_pred"] = ["42", "42", "6"]
# │ example["all_eval"] = [True, True, False]
# │ example["mv_pred"] = "42"
# │ example["mv_eval"] = True
# │ example["mv_index"] = 0   