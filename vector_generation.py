import torch
import os
import argparse


def load_data(data_dir, prefixs, layer_num=29, max_examples=None):
    """
    从指定目录加载数据，并将数据分为 check, switch, other 三个类别
    Args:
        data_dir (str): 数据存放的目录路径。
        prefixs (list[str]): 前缀列表，用于构建数据路径。
        layer_num (int, optional): 层数，默认为29。
        max_examples (int, optional): 最多加载的样本数，如果为None则加载全部样本。
    Returns:
        tuple: 包含check, switch, other (torch.Tensor)
    """
    data_paths = [os.path.join(data_dir, f"hidden_{p}", "hidden.pt") for p in prefixs]
    switch = [[] for _ in range(layer_num)]
    check = [[] for _ in range(layer_num)]
    other = [[] for _ in range(layer_num)]
    for i, data_path in enumerate(data_paths):
        data = torch.load(data_path, weights_only=False) # 数组
        # 遍历每一层、每个样本，提取：所有 step 的隐藏状态 反思/转换 的位置索引
        for l in range(layer_num):
            layer_data = data[l] # 某一层的所有样本 {sample_id: {}...}
            for k in layer_data: # 数字
                if max_examples is not None and max_examples > 0 and k >= max_examples:
                    continue
                h = layer_data[k]["step"] # torch.Size([28, 1536]) # 28个step
                check_index = layer_data[k]["check_index"] # tensor([19, 24]) shape(2)
                switch_index = layer_data[k]["switch_index"] # tensor([25]) shape(1)
                check[l].append(h[check_index])
                switch[l].append(h[switch_index])
                all_indices = torch.arange(h.shape[0])
                mask = ~(torch.isin(all_indices, check_index) | torch.isin(all_indices, switch_index))
                other[l].append(h[mask])
    for l in range(layer_num):
        check[l] = torch.cat(check[l], dim=0)
        switch[l] = torch.cat(switch[l], dim=0)
        other[l] = torch.cat(other[l], dim=0)
    check = torch.stack(check, dim=0)
    switch = torch.stack(switch, dim=0)
    other = torch.stack(other, dim=0)
    # torch.Size: ([21, 212, 1536]), ([21, 66, 1536]), ([21, 1239, 1536])
    return check, switch, other

# 根据 load_data 生成向量并保存为pt
def generate_vector_switch_check(data_dir, prefixs, layers, save_prefix, overwrite=False):
    if isinstance(layers, int):
        layers = [layers]
    max_layer = max(layers)
    check, switch, other = load_data(data_dir=data_dir, prefixs=prefixs, layer_num=max_layer+1)
    save_dir = os.path.join(data_dir, f"vector_{save_prefix}")
    print(f"save_dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        layer_check = check[layer]
        layer_switch = switch[layer]
        layer_other = other[layer]
        steer_vec = torch.cat([layer_check, layer_switch], dim=0).mean(dim=0) - layer_other.mean(dim=0)
        save_path = os.path.join(save_dir, f"layer_{layer}_transition_reflection_steervec.pt")
        if not os.path.exists(save_path) or overwrite:
            torch.save(steer_vec, save_path)
        else:
            print(f"{save_path} already exists")
        print(f"layer {layer} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--prefixs", type=str, nargs="+", default=["correct_0_500", "incorrect_0_500"])
    parser.add_argument("--layers", type=int, nargs="+", default=[20])
    parser.add_argument("--save_prefix", type=str, default="500_500")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_vector_switch_check(
        data_dir=args.data_dir,
        prefixs=args.prefixs,
        layers=args.layers,
        save_prefix=args.save_prefix,
        overwrite=args.overwrite
    )
    # generate_vector_switch_check(
    #     data_dir="results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000",
    #     prefixs=["correct_0_500", "incorrect_0_500"],
    #     layers=20,
    #     save_prefix="500_500",
    # )