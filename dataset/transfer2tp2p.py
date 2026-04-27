import json

input_path = "/home/ec2-user/GenAI-Project/data/stylebooth_dataset/train.csv"
output_path = "/home/ec2-user/GenAI-Project/data/stylebooth_dataset/metadata.jsonl"

count = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    # 跳过表头
    header = fin.readline().rstrip("\n")
    print("Header:", header)

    for line_num, line in enumerate(fin, start=2):
        line = line.rstrip("\n")
        if not line.strip():
            continue

        # 只分割前 5 个逗号，这样最后一列 dict 整体保留
        parts = line.split(",", 5)

        if len(parts) < 5:
            print(f"Skipping bad line {line_num}: {line}")
            continue

        # 前6列分别是：
        # 0 EnStyle
        # 1 ZhStyle
        # 2 ShortStyleName
        # 3 Target:FILE
        # 4 Source:FILE
        # 5 SearchResult:DICT  (可有可无，我们这里不用)
        enstyle = parts[0].strip()
        target_file = parts[3].strip()
        source_file = parts[4].strip()

        record = {
            "input_image_file_name": source_file,
            "edited_image_file_name": target_file,
            "edit_prompt": f"turn it into {enstyle}"
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Saved {count} rows to {output_path}")