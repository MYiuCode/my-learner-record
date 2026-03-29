import os
import sys
import subprocess
import json
import shutil

# ==========================================
# 1. 配置路径
# ==========================================
RAW_DATA_PATH = r"D:\nnUNet_raw"
PREPROCESSED_PATH = r"D:\nnUNet_preprocessed"
RESULTS_PATH = r"D:\nnUNet_results"
DATASET_ID = "666"
DATASET_NAME = f"Dataset{DATASET_ID}_MRIBrainHM"

# ==========================================
# 2. 强制修复 dataset.json
# ==========================================
json_path = os.path.join(RAW_DATA_PATH, DATASET_NAME, "dataset.json")

correct_data = {
    "channel_names": {
        "0": "cineMRI"
    },
    "labels": {
        "background": 0,
        "hippocampus_L": 1,
        "hippocampus_R": 2
    },
    "numTraining": 58,
    "file_ending": ".nii.gz"
}

print(f"🔧 正在强制修复/重写 dataset.json...")
try:
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(correct_data, f, indent=4)
    print(f"✅ dataset.json 已强制修复成功！")
except Exception as e:
    print(f"❌ 修复失败: {e}")
    sys.exit(1)

# ==========================================
# 3. 设置环境变量
# ==========================================
print(f"🚀 正在设置 nnUNet 环境变量...")
os.environ['nnUNet_raw'] = RAW_DATA_PATH
os.environ['nnUNet_preprocessed'] = PREPROCESSED_PATH
os.environ['nnUNet_results'] = RESULTS_PATH

print(f"✅ 环境变量设置成功:")
print(f"   - Raw: {os.environ['nnUNet_raw']}")
print(f"   - Preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"   - Results: {os.environ['nnUNet_results']}")

# ==========================================
# 4. 自动清理旧数据 (可选，防止配置冲突)
# ==========================================
preprocessed_dataset_folder = os.path.join(PREPROCESSED_PATH, DATASET_NAME)
if os.path.exists(preprocessed_dataset_folder):
    print(f"\n🧹 检测到旧预处理数据，正在清理...")
    try:
        shutil.rmtree(preprocessed_dataset_folder)
        print("✅ 清理完成！")
    except Exception as e:
        print(f"⚠️ 清理失败，请手动删除该文件夹: {e}")
else:
    print(f"\n🆕 准备开始新的预处理。")

# ==========================================
# 5. 执行命令
# ==========================================
def run_command(cmd, description):
    print(f"\n--- ⏳ 正在执行: {description} ---")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} 完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败！错误信息: {e}")
        sys.exit(1)

# --- 步骤 A: 规划与预处理 ---
# 使用默认规划器
cmd_preprocess = [
    "nnUNetv2_plan_and_preprocess",
    "-d", DATASET_ID,
    "--verify_dataset_integrity"
]
run_command(cmd_preprocess, "步骤 A: 规划与预处理")

# --- 步骤 B: 训练模型 ---
# 修正：去掉了所有过时参数 (-np, --fp16)，使用最纯净的命令
cmd_train = [
    "nnUNetv2_train",
    DATASET_ID,
    "2d",
    "0"
]
run_command(cmd_train, "步骤 B: 训练模型 (2D)")

print("\n🎉 所有任务执行完毕！")