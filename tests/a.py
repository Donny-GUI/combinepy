import argparse
import ast
import os

def extract_imports_with_ast(file_path):
    """
    Extracts import statements and code blocks from a Python file using AST.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    imports = []
    other_code = []
    
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        else:
            other_code.append(node)
    
    return imports, other_code

def imports_to_code(imports):
    """
    Converts a list of AST import nodes back into Python code.
    """
    return "\n".join(ast.unparse(imp) for imp in imports)

def combine_files_with_ast(file_paths, output_file):
    """
    Combines multiple Python files using AST, deduplicates imports, and writes to an output file.
    """
    all_imports = []
    all_other_code = []
    
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue
        
        imports, other_code = extract_imports_with_ast(file_path)
        all_imports.extend(imports)
        all_other_code.extend(other_code)
    
    # Deduplicate imports by converting them to strings and back to AST nodes
    unique_imports = list({ast.dump(imp): imp for imp in all_imports}.values())
    
    # Write combined code to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        # Write imports
        if unique_imports:
            f.write(imports_to_code(unique_imports) + "\n\n")
        # Write other code
        for node in all_other_code:
            f.write(ast.unparse(node) + "\n\n")
    
    print(f"Files combined into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine Python files with organized imports.")
    parser.add_argument("files", nargs="+", help="Python files to combine.")
    parser.add_argument("-o", "--output", default="combined.py", help="Output file (default: combined.py).")
    args = parser.parse_args()
    
    combine_files_with_ast(args.files, args.output)

if __name__ == "__main__":
    main()
