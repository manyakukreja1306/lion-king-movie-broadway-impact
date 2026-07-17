import json

def extract_and_run():
    with open("Data-analysis-initial.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]

    with open("run_data_analysis.py", "w", encoding="utf-8") as f:
        for cell in code_cells:
            if isinstance(cell, list):
                # filter out magic commands like %matplotlib inline
                code = "".join([line for line in cell if not line.startswith("%") and not line.startswith("!")])
                f.write(code + "\n\n")
            else:
                if not cell.startswith("%") and not cell.startswith("!"):
                    f.write(cell + "\n\n")

if __name__ == "__main__":
    extract_and_run()
