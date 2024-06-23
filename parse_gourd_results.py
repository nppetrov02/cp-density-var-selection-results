import re
import os
import argparse
import json

def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the file name from the first line
    file_name = lines[0].strip().split("file name: ")[1][1: -1]

    # Join the rest of the lines for the content parsing
    content = ''.join(lines[1:])

    # Extract general information
    general_info_pattern = re.compile(r"jobs\s*=\s*(\d+);\s*secondary\s*resources\s*=\s*(\d+)")
    general_info_match = general_info_pattern.search(content)
    jobs = int(general_info_match.group(1))
    secondary_resources = int(general_info_match.group(2))

    # Extract statistics
    stat_blocks_pattern = re.compile(r"%%%mzn-stat: objective=(\d+)\n(.*?)%%%mzn-stat-end", re.DOTALL)
    stat_blocks = stat_blocks_pattern.findall(content)

    stats = []
    for block in stat_blocks:
        objective = int(block[0])
        stats_lines = block[1].strip().split("\n")
        stat_dict = {"objective": objective}
        for line in stats_lines:
            key, value = line.replace("%%%mzn-stat: ", "").split("=")
            key = re.sub(r'([A-Z])', r'_\1', key).lower()  # Convert camelCase to snake_case
            stat_dict[key] = float(value) if '.' in value else int(value)
        stats.append(stat_dict)

    # Extract status
    status_pattern = re.compile(r"status:\s*(\w+)\s*(\d+)\s*(\d+)")
    status_match = status_pattern.search(content)
    if ( not status_match):
        print(file_path)
    if status_match:
        status = status_match.group(1)
        final_objective = int(status_match.group(2))
        final_time = int(status_match.group(3))
    else:
        status_pattern = re.compile(r"status:\s*(\w+)")
        status_match = status_pattern.search(content)
        status = status_match.group(1)
        final_objective = None
        final_time = None

    result = {
        "file_name": file_name,
        "jobs": jobs,
        "secondary_resources": secondary_resources,
        "stats": stats,
        "status": {
            "status": status,
            "final_objective": final_objective,
            "final_time": final_time
        }
    }

    return result

def parse_directory(directory_path):
    all_data = []

    # Iterate over each folder in the directory
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            log_file_path = os.path.join(folder_path, "run.log")

            # Check if run.log exists in the folder
            if os.path.exists(log_file_path):
                # Parse run.log and collect data
                parsed_data = parse_file(log_file_path)
                all_data.append(parsed_data)

    return all_data

def save_to_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse run.log files in a directory and save data to JSON.")
    parser.add_argument("--directory_path", type=str, required=True, help="Path to the directory containing run.log files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    directory_path = args.directory_path.strip()
    output_file = args.output_file.strip()

    # Validate directory path existence
    if os.path.isdir(directory_path):
        parsed_data_list = parse_directory(directory_path)

        # Save parsed data to JSON file
        save_to_json(parsed_data_list, output_file)
        
        print(f"Data saved to {output_file}")
    else:
        print(f"Error: Directory '{directory_path}' does not exist.")