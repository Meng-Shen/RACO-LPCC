#!/usr/bin/env python3
import argparse
import csv
import re


CLASS_HEADERS = {
    "Car": r"Car\s+AP_R40@0\.70,\s*0\.70,\s*0\.70",
    "Pedestrian": r"Pedestrian\s+AP_R40@0\.50,\s*0\.50,\s*0\.50",
    "Cyclist": r"Cyclist\s+AP_R40@0\.50,\s*0\.50,\s*0\.50",
}


def parse_log(path):
    row = {
        "Car_3d_AP_R40_easy": "",
        "Car_3d_AP_R40_moderate": "",
        "Car_3d_AP_R40_hard": "",
        "Pedestrian_3d_AP_R40_easy": "",
        "Pedestrian_3d_AP_R40_moderate": "",
        "Pedestrian_3d_AP_R40_hard": "",
        "Cyclist_3d_AP_R40_easy": "",
        "Cyclist_3d_AP_R40_moderate": "",
        "Cyclist_3d_AP_R40_hard": "",
    }
    waiting_class = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            for cls, pattern in CLASS_HEADERS.items():
                if re.search(pattern, line):
                    waiting_class = cls
                    break
            else:
                if waiting_class and line.startswith("3d") and "AP:" in line:
                    values = [float(x) for x in re.findall(r"[0-9]+(?:\.[0-9]+)?", line.split("AP:", 1)[1])]
                    if len(values) >= 3:
                        row[f"{waiting_class}_3d_AP_R40_easy"] = values[0]
                        row[f"{waiting_class}_3d_AP_R40_moderate"] = values[1]
                        row[f"{waiting_class}_3d_AP_R40_hard"] = values[2]
                    waiting_class = None
    return row


def main():
    parser = argparse.ArgumentParser(description="Parse one adaptive JUCP AP eval log into one curve row.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rate_id", required=True, type=int)
    parser.add_argument("--threshold", default="")
    parser.add_argument("--label_csv", default="")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    row = parse_log(args.log)
    row = {"rate_id": args.rate_id, "threshold": args.threshold, "label_csv": args.label_csv, **row}
    fieldnames = list(row.keys())
    mode = "a" if args.append else "w"
    write_header = not args.append
    with open(args.out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Parsed adaptive AP row rate_id={args.rate_id} -> {args.out}")


if __name__ == "__main__":
    main()
