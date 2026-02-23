import os, re

def update_headers_cpp(header, header_pattern, root_dir, exts=(".hpp", ".cpp", ".h", ".c")):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(exts):
                path = os.path.join(root, f)
                if "build" in path:
                    continue

                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()

                if header_pattern.match(content):
                    new_content = header_pattern.sub(header, content, count=1)
                else:
                    new_content = header + "\n\n" + content

                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content) 

                print(f'Wrote header to {path}')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = script_dir + '/../../'
    file_path = script_dir + '/header_proprietary_cpp.txt'
    file_contents = ''
    header_pattern = re.compile(r"\/\*\s*(.*?)\s*\*\/", re.DOTALL)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    update_headers_cpp(file_contents, header_pattern, root)
