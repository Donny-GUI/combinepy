import argparse
import ast
import os
import sys
from rich.status import Status
from .visitor import CombinerVisitor, CombineWriter
from app.feature import FeatureApplier


def get_python_files_from_directory(directory:str):
    return [
        os.path.join(directory, file) 
        for file in os.listdir(directory) 
        if file.endswith(".py")
        ]


def remove_any_relative_imports(source:str):
    lines = source.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('from .'):
            continue
        new_lines.append(line)
    return '\n'.join(new_lines)

def gap_after_imports(source:str):
    lines = source.split('\n')
    new_lines = []
    last = ""
    for line in lines:
        if last.startswith('import ') and not line.startswith('import '):
            new_lines.append('')
        new_lines.append(line)
        last = line

    return '\n'.join(new_lines)


def dir_is_module(directory: str):
    return os.path.isfile(os.path.join(directory, "__init__.py"))

def combine_files_with_ast(file_paths:list[str], output_file: str):
    """
    Combines multiple Python files using AST, deduplicates imports, and writes to an output file.
    
    This function takes a list of paths to Python files and directories, and an output file path.
    It returns nothing, but writes the combined output to the specified file.
    """
    with Status("Combining files...") as status:

    
        # Take a copy of the list, because we're going to modify it in place
        x = file_paths
        file_paths = []

        # This list will contain the names of the modules that we're combining
        mods = []

        status.update("Gathering files...") 
        # Iterate over the list of file paths
        for file_path in x:
            # If the path is a directory, get a list of all the python files in it
            # and add them to the list of files to be processed
            if os.path.isdir(file_path):

                # Get a list of all the python files in the directory
                files = get_python_files_from_directory(file_path)
                
                # If the directory contains an __init__.py file, then it's a module
                # so add the name of the module to the list of modules
                if dir_is_module(file_path):
                    mods.append(os.path.basename(file_path))
                    for f in files:
                        mods.append(os.path.join(file_path,os.path.splitext(f)[0]))
                
                # Add the files to the list of files to be processed
                file_paths.extend(files)

            # If the path is a file, just add it to the list of files to be processed
            else:
                file_paths.append(file_path)

        # If there are no files to process, then print a message and exit
        if len(file_paths) == 0:
            print("No files to combine. Exiting.")
            sys.exit(0)
        

        status.update("Combining files...")


        # Create a visitor that will visit all the files and combine their contents
        combine = CombinerVisitor()
        c = 0
        # Iterate over all the files and visit them
        for file_path in file_paths:
            
            # Skip any files that don't exist
            if not os.path.isfile(file_path):
                print(f"Warning: {file_path} does not exist. Skipping.")
                continue

            # Print a message to indicate which file is being processed
            c += 1
            print(f"Processing file: {file_path}")
            # Open the file and parse it using the AST
            with open(file_path, "r", encoding="utf-8") as f:
                structure = ast.parse(f.read())
            # Visit the file using the visitor
            combine.visit(structure)

        # If there are no files to combine, print a message and exit
        if c == 0:
            print("No files to combine. Exiting.")
            return

        # Create a writer that will write out the combined source code
        writer = CombineWriter(combine)
        # Write out the combined source code
        output = writer.string()

        status.update("Applying features...")

        # Apply some features to the combined source code
        features = FeatureApplier(output)
        features.remove_unused_imports()
        features.remove_duplicate_classes()
        features.remove_duplicate_functions()
        features.put_nameismain_last()
        features.class_assigns_before_any_method()
        features.private_methods_first()
        features.alphabetize_imports()
        features.group_constants_together()
        features.main_function_is_last()
        features.remove_duplicate_constants()
        features.format_with_autopep8()

        source = features.source

        status.update("Writing output...")

        source = remove_any_relative_imports(source)
        source = gap_after_imports(source)

        # Write the final combined source code to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            # Write imports
            f.write(source)
            f.write("\n")
        
        print(f"Files combined into {output_file}")


def run():
    
    parser = argparse.ArgumentParser(description="Combine Python source files.",
                                     usage="combine [options] <file1.py> <file2.py> <directory>...",
                                     formatter_class=argparse.RawTextHelpFormatter
                                     )
    
    parser.add_argument("files", 
                        nargs="+", 
                        help="Python files or directories to combine.")
    
    parser.add_argument("-o", "--output", 
                        default="combined.py", 
                        help="Output file (default: combined.py).")
    
    args = parser.parse_args()

    combine_files_with_ast(args.files, args.output)

