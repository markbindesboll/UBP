import sys
import os
import torch

DEFAULT_PATH = '/data/thingseeg2/eeg_preprocessed_250hz/ubp_format/Image_feature/FoveaBlur/RN50_ua_k51_g3_c6_test.pt'
target_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

print("=" * 80)
print(f"Deep Inspection: {target_path}")
print("=" * 80)

data = torch.load(target_path, map_location='cpu', weights_only=False)

# -------------------------------------------------------------
# 1. Inspect 'text_features'
# -------------------------------------------------------------
if 'text_features' in data:
    tf = data['text_features']
    print(f"\n[1] 'text_features' Breakdown:")
    print(f"    Type: {type(tf).__name__} | Total Concepts: {len(tf)}")
    
    tf_keys = list(tf.keys())
    sample_key = tf_keys[0]
    sample_val = tf[sample_key]
    
    if hasattr(sample_val, 'shape'):
        print(f"    Embedding Tensor Shape : {tuple(sample_val.shape)} | Dtype: {sample_val.dtype}")
    
    print("\n    Sample Concept Embeddings (first 5):")
    for k in tf_keys[:5]:
        v = tf[k]
        shape_info = f"shape {tuple(v.shape)}" if hasattr(v, 'shape') else str(type(v))
        print(f"      - '{k}': {shape_info}")

# -------------------------------------------------------------
# 2. Inspect 'img_features'
# -------------------------------------------------------------
if 'img_features' in data:
    img_f = data['img_features']
    print(f"\n[2] 'img_features' Breakdown:")
    print(f"    Type: {type(img_f).__name__} | Levels / Sub-keys: {list(img_f.keys())}")
    
    for level, sub_dict in img_f.items():
        print(f"\n    --- Level: '{level}' ---")
        if isinstance(sub_dict, dict):
            paths = list(sub_dict.keys())
            print(f"    Total Images: {len(paths)}")
            
            if paths:
                first_path = paths[0]
                first_tensor = sub_dict[first_path]
                if hasattr(first_tensor, 'shape'):
                    print(f"    Feature Tensor Shape : {tuple(first_tensor.shape)} | Dtype: {first_tensor.dtype}")
                
                print("    Sample Image Paths (first 3):")
                for p in paths[:3]:
                    t = sub_dict[p]
                    shape_info = f"shape {tuple(t.shape)}" if hasattr(t, 'shape') else str(type(t))
                    print(f"      * {p} -> {shape_info}")
        else:
            print(f"    Value Type: {type(sub_dict).__name__} | Content: {sub_dict}")

print("\n" + "=" * 80)

super_cat_path = '/data/thingseeg2/images/categories/test_super_cat_list.pth'
cat_path = '/data/thingseeg2/images/categories/test_cat_list.pth'

super_cats = torch.load(super_cat_path, map_location='cpu', weights_only=False)
cats = torch.load(cat_path, map_location='cpu', weights_only=False)

print(f"{'Idx':<4} | {'Category (test_cat)':<30} | {'Super-Category (test_super_cat)':<30}")
print("-" * 70)

for i in range(min(50, len(cats), len(super_cats))):
    print(f"{i+1:<4} | {str(cats[i]):<30} | {str(super_cats[i]):<30}")

from collections import Counter
cat_counts = Counter(cats)
super_cat_counts = Counter(super_cats)

print("=" * 65)
print(f"Super Categories (test_super_cat) — Total Unique: {len(super_cat_counts)} | Total Trials: {len(super_cats)}")
print("=" * 65)
print(f"{'Super Category':<40} | {'Trials':>6} | {'% of Test Set':>12}")
print("-" * 65)
for cat, count in super_cat_counts.most_common():
    print(f"{str(cat):<40} | {count:>6} | {count/len(super_cats)*100:>11.1f}%")

print("\n" + "=" * 65)
print(f"Standard Categories (test_cat) — Total Unique: {len(cat_counts)} | Total Trials: {len(cats)}")
print("=" * 65)
print(f"{'Category':<40} | {'Trials':>6} | {'% of Test Set':>12}")
print("-" * 65)
for cat, count in cat_counts.most_common():
    print(f"{str(cat):<40} | {count:>6} | {count/len(cats)*100:>11.1f}%")
print("=" * 65)

path = '/data/thingseeg2/ubp_exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50/sub-08_seed0/test_embeddings.pt'

print("=" * 80)
print(f"Inspecting: {path}")
print("=" * 80)

if not os.path.exists(path):
    print(f"Error: Path does not exist -> {path}")
else:
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Root Object Type: {type(data).__name__}\n")

    if isinstance(data, dict):
        print(f"Total Keys: {len(data)}")
        print("-" * 80)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"Key: {str(k):<15} | Tensor | Shape: {tuple(v.shape)} | Dtype: {v.dtype}")
            elif isinstance(v, (list, tuple)):
                sample_item = f" (first: {repr(v[0])[:40]})" if len(v) > 0 else ""
                print(f"Key: {str(k):<15} | {type(v).__name__:<6} | Length: {len(v)}{sample_item}")
            else:
                print(f"Key: {str(k):<15} | {type(v).__name__:<6} | Value: {repr(v)}")

    elif isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            first = data[0]
            print(f"First item type: {type(first).__name__}")
            if hasattr(first, 'shape'):
                print(f"First item shape: {tuple(first.shape)}")
            elif isinstance(first, dict):
                print(f"First item keys: {list(first.keys())}")
    else:
        print(data)