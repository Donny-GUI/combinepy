from app.feature import FeatureApplier
from ast import *
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum, auto, Enum, _auto_null, _is_dunder, _is_descriptor, _is_sunder, Flag, _is_private, _is_single_bit
from io import StringIO
from rich.status import Status
from typing import Optional
import argparse
import ast
import autopep8
import inspect
import os
import shutil
import subprocess
import sys
import tokenize

FATAL = '❌'
WARN = '⛔'
GOOD = '✅'
CONV = '🆗'
RECM = '❎'
WALL = '|'
DIMBLACK = '\x1b[2;30m'
_INFSTR = '1e' + repr(sys.float_info.max_10_exp + 1)
_SINGLE_QUOTES = ("'", '"')
_MULTI_QUOTES = ('"""', "'''")
_ALL_QUOTES = (*_SINGLE_QUOTES, *_MULTI_QUOTES)


def get_python_files_from_directory(directory: str):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.py')]


def remove_any_relative_imports(source: str):
    lines = source.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('from .'):
            continue
        new_lines.append(line)
    return '\n'.join(new_lines)


def gap_after_imports(source: str):
    lines = source.split('\n')
    new_lines = []
    last = ''
    for line in lines:
        if last.startswith('import ') and (not line.startswith('import ')):
            new_lines.append('')
        new_lines.append(line)
        last = line
    return '\n'.join(new_lines)


def dir_is_module(directory: str):
    return os.path.isfile(os.path.join(directory, '__init__.py'))


def combine_files_with_ast(file_paths: list[str], output_file: str):
    """
    Combines multiple Python files using AST, deduplicates imports, and writes to an output file.

    This function takes a list of paths to Python files and directories, and an output file path.
    It returns nothing, but writes the combined output to the specified file.
    """
    with Status('Combining files...') as status:
        x = file_paths
        file_paths = []
        mods = []
        status.update('Gathering files...')
        for file_path in x:
            if os.path.isdir(file_path):
                files = get_python_files_from_directory(file_path)
                if dir_is_module(file_path):
                    mods.append(os.path.basename(file_path))
                    for f in files:
                        mods.append(os.path.join(
                            file_path, os.path.splitext(f)[0]))
                file_paths.extend(files)
            else:
                file_paths.append(file_path)
        if len(file_paths) == 0:
            print('No files to combine. Exiting.')
            sys.exit(0)
        status.update('Combining files...')
        combine = CombinerVisitor()
        c = 0
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                print(f'Warning: {file_path} does not exist. Skipping.')
                continue
            c += 1
            print(f'Processing file: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as f:
                structure = ast.parse(f.read())
            combine.visit(structure)
        if c == 0:
            print('No files to combine. Exiting.')
            return
        writer = CombineWriter(combine)
        output = writer.string()
        status.update('Applying features...')
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
        status.update('Writing output...')
        source = remove_any_relative_imports(source)
        source = gap_after_imports(source)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(source)
            f.write('\n')
        print(f'Files combined into {output_file}')


def run():
    parser = argparse.ArgumentParser(description='Combine Python source files.',
                                     usage='combine [options] <file1.py> <file2.py> <directory>...', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('files', nargs='+',
                        help='Python files or directories to combine.')
    parser.add_argument('-o', '--output', default='combined.py',
                        help='Output file (default: combined.py).')
    args = parser.parse_args()
    combine_files_with_ast(args.files, args.output)


class ClassEnum:
    """
    A utility class to enumerate methods and attributes of a given class.
    """

    @classmethod
    def getAttributes(cls, target_class) -> list:
        """
        Retrieves all attributes (both descriptors and non-descriptors) of a given class.

        Args:
            target_class (type): The class to inspect.

        Returns:
            list: A list of attribute names.
        """
        attributes = [attr for attr, _ in inspect.getmembers(
            target_class, predicate=inspect.isdatadescriptor)]
        non_descriptor_attributes = [attr for attr, value in inspect.getmembers(
            target_class) if not callable(value) and (not attr.startswith('__'))]
        return attributes + non_descriptor_attributes

    @classmethod
    def getMethods(cls, target_class) -> dict:
        """
        Retrieves all callable methods of a given class, categorized as:
        - Instance methods
        - Class methods
        - Static methods

        Args:
            target_class (type): The class to inspect.

        Returns:
            dict: A dictionary with keys 'instance_methods', 'class_methods', and 'static_methods',
                  each containing a list of corresponding method names.
        """
        instance_methods = [meth for meth, func in inspect.getmembers(
            target_class, predicate=inspect.isfunction)]
        all_methods = inspect.getmembers(
            target_class, predicate=inspect.ismethod)
        class_methods = [meth for meth, m in all_methods if isinstance(
            m.__func__, classmethod)]
        static_methods = [meth for meth, m in inspect.getmembers(
            target_class) if isinstance(m, staticmethod)]
        return {'instance_methods': instance_methods, 'class_methods': class_methods, 'static_methods': static_methods}


def is_if_name_is_main(stmt: ast.If):
    string = ast.unparse(stmt.test)
    items = ['__name__', '__main__', '==']
    for item in items:
        if item not in string:
            return False
    return True


class FeatureApplier:

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self._format = None

    def _rewriteSource(self):
        """Updates the source code string from the current AST."""
        self.source = ast.unparse(self.tree)
        match self._format:
            case None:
                return
            case 'autopep8':
                self.format_with_autopep8()
            case 'black':
                self.format_with_black()

    def _rewriteTree(self):
        """Parses the current source code into an abstract syntax tree (AST) and updates the tree attribute."""
        self.tree = ast.parse(self.source)

    def _body(self, body: list[ast.AST]):
        """
        Sets the body of the current AST to the given list of nodes. 
        This is a lower-level version of the other methods in this class, 
        which create nodes or modify the tree for you.
        """
        setattr(self.tree, 'body', body)

    def remove_unused_imports(self):
        """
        Removes any unused imports from the current source code.

        This method iterates over the entire AST and stores the names of all
        imports in a dictionary, mapping the name to the line number of the
        import.  It then iterates over the entire AST again and stores the names
        of any names used in the code in a set.  Any imports that are not used
        are stored in a dictionary, mapping the name to the line number of the
        import.  The body of the current AST is then updated to exclude any
        unused imports.  Finally, the source code string is updated from the
        current AST.
        """
        imports = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name or alias.asname] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                module = f'{node.module}.' if node.module else ''
                for alias in node.names:
                    full_name = f'{module}{alias.name or alias.asname}'
                    imports[full_name] = node.lineno
        used_names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            if isinstance(node, ast.ClassDef) and node.bases:
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        used_names.add(base.id)

        def name_contained_in_module(name: str, module: str) -> bool:
            return module.find(name) != -1
        unused = {name: lineno for name,
                  lineno in imports.items() if name not in used_names}
        for name in used_names:
            kick = []
            for notused in unused.keys():
                if notused.find(name) != -1:
                    kick.append(notused)
            for k in kick:
                del unused[k]
        keeping = []
        for node in self.tree.body:
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if node.lineno not in unused.values():
                    keeping.append(node)
            else:
                keeping.append(node)
        self._body(keeping)
        self._rewriteSource()

    def big_lists_one_per_line(self):
        """
        takes lists that are long and puts each item on its own line
        """
        pass

    def class_assigns_before_any_method(self):
        """
        Reorders class assignments to appear before any methods in the class body.

        This function ensures that all assignments within a class definition are 
        placed before any method definitions. This can help in maintaining a 
        consistent structure within the class, making it easier to understand the 
        class's attributes and their initial values before diving into the methods.
        """

        def rearrange_class_assignments(node: ast.ClassDef):
            b: list[ast.AST] = node.body
            x = []
            for i, item in enumerate(b):
                if isinstance(item, ast.Assign):
                    x.insert(0, item)
                else:
                    x.append(item)
            setattr(node, 'body', x)
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                rearrange_class_assignments(node)
        self._rewriteSource()

    def alphabetize_methods(self):
        pass

    def _name_is_constant(self, name: ast.Name):
        return name.id.isupper()

    def _assign_is_constant(self, node: ast.Assign):
        return len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0])

    def format_with_autopep8(self):
        self.source = autopep8.fix_code(self.source)
        self.tree = ast.parse(self.source)
        self._format = 'autopep8'

    def alphabetize_imports(self):
        pass

    def put_nameismain_last(self):
        """Moves any if __name__ == "__main__": blocks to the end of the file.

        This is done by iterating over the body of the current AST, and 
        storing any non-if __name__ == "__main__": blocks in a list. 
        If any if __name__ == "__main__": blocks are found, they are stored 
        in a separate list. If no if __name__ == "__main__": blocks are found, 
        this method does nothing. If any are found, they are appended to the 
        end of the list of non-if __name__ == "__main__": blocks and the body 
        of the current AST is updated to this new list. The source code string 
        is then updated from the current AST.
        """
        ifname_body = []
        keeping = []
        has_main = False
        for node in self.tree.body:
            if isinstance(node, ast.If) and is_if_name_is_main(node):
                ifname_body.extend(node.body)
                has_main = True
            else:
                keeping.append(node)
        if has_main == False:
            return
        ifn = ast.If(test=ast.Compare(left=ast.Name(id='__name__', ctx=ast.Load()), ops=[
                     ast.Eq()], comparators=[ast.Constant('__main__')]), body=ifname_body)
        keeping.append(ifn)
        self._body(keeping)
        self._rewriteSource()

    def main_function_is_last(self):
        mainfunc = None
        lastindex = 0
        c = 0
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                lastindex = c
                if node.name == 'main':
                    mainfunc = node
            c += 1
        if mainfunc != None:
            self.tree.body.remove(mainfunc)
            self.tree.body.insert(lastindex, mainfunc)
            self._rewriteSource()

    def group_constants_together(self):
        """
        Groups all constant assignments (assignments to uppercase names) together

        This function will go through the Abstract Syntax Tree (AST) and find the first constant assignment. 
        It will then group all following constant assignments together, and move them all to the front of the file.
        """
        first_constant = len(self.tree.body)
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
                    first_constant = self.tree.body.index(node)
                    break
        cons = []
        for node in self.tree.body[first_constant:]:
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
                    cons.append(node)
                else:
                    break
            else:
                break
        self.tree.body = self.tree.body[:first_constant] + \
            cons + self.tree.body[first_constant:]
        self._rewriteSource()

    def private_methods_first(self):
        """
        for classes, write private methods first before public methods
        """
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        if subnode.name.startswith('_') == False:
                            node.body.remove(subnode)
                            node.body.insert(-1, subnode)
        self._rewriteSource()


def resolve_relative_imports(ast_module: ast.Module, base_module: str) -> str:
    """
    Converts relative imports in an AST module to absolute imports and appends them to the source.

    Args:
        ast_module (ast.Module): The AST of the Python module.
        base_module (str): The base module path (e.g., 'package.subpackage.module').

    Returns:
        str: The modified source code with absolute imports added.
    """
    relative_imports = []

    class RelativeImportFinder(ast.NodeVisitor):

        def visit_ImportFrom(self, node):
            if node.level > 0:
                base_parts = base_module.split('.')
                absolute_path = base_parts[:-node.level]
                if node.module:
                    absolute_path.append(node.module)
                absolute_import = f'from {'.'.join(absolute_path)} import {', '.join((a.name for a in node.names))}'
                relative_imports.append(absolute_import)
            self.generic_visit(node)
    finder = RelativeImportFinder()
    finder.visit(ast_module)
    original_source = ast.unparse(ast_module)
    resolved_imports = '\n'.join(relative_imports)
    if resolved_imports:
        return f'{original_source}\n# Resolved Relative Imports:\n{resolved_imports}'
    else:
        return original_source


def is_if_name_is_main(node):
    string = ast.unparse(node.test)
    items = ['__name__', '__main__', '==']
    for item in items:
        if item not in string:
            return False
    return True


class FeatureApplier:

    def __init__(self, source: str):
        self.source: str = source
        self.tree = ast.parse(source)
        self._format: str = None

    def _rewriteSource(self):
        """Updates the source code string from the current AST."""
        self.source = ast.unparse(self.tree)
        match self._format:
            case None:
                return
            case 'autopep8':
                self.format_with_autopep8()
            case 'black':
                self.format_with_black()

    def _rewriteTree(self):
        """Parses the current source code into an abstract syntax tree (AST) and updates the tree attribute."""
        self.tree = ast.parse(self.source)

    def _body(self, body: list[ast.AST]):
        """
        Sets the body of the current AST to the given list of nodes. 
        This is a lower-level version of the other methods in this class, 
        which create nodes or modify the tree for you.
        """
        setattr(self.tree, 'body', body)

    def remove_unused_imports(self):
        """
        Removes any unused imports from the current source code.

        This method iterates over the entire AST and stores the names of all
        imports in a dictionary, mapping the name to the line number of the
        import.  It then iterates over the entire AST again and stores the names
        of any names used in the code in a set.  Any imports that are not used
        are stored in a dictionary, mapping the name to the line number of the
        import.  The body of the current AST is then updated to exclude any
        unused imports.  Finally, the source code string is updated from the
        current AST.
        """
        imports = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name or alias.asname] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                module = f'{node.module}.' if node.module else ''
                for alias in node.names:
                    full_name = f'{module}{alias.name or alias.asname}'
                    imports[full_name] = node.lineno
        used_names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            if isinstance(node, ast.ClassDef) and node.bases:
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        used_names.add(base.id)

        def name_contained_in_module(name: str, module: str) -> bool:
            return module.find(name) != -1
        unused = {name: lineno for name,
                  lineno in imports.items() if name not in used_names}
        for name in used_names:
            kick = []
            for notused in unused.keys():
                if notused.find(name) != -1:
                    kick.append(notused)
            for k in kick:
                del unused[k]
        keeping = []
        for node in self.tree.body:
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if node.lineno not in unused.values():
                    keeping.append(node)
            else:
                keeping.append(node)
        self._body(keeping)
        self._rewriteSource()

    def big_lists_one_per_line(self):
        """
        takes lists that are long and puts each item on its own line
        """
        pass

    def class_assigns_before_any_method(self):
        """
        Reorders class assignments to appear before any methods in the class body.

        This function ensures that all assignments within a class definition are 
        placed before any method definitions. This can help in maintaining a 
        consistent structure within the class, making it easier to understand the 
        class's attributes and their initial values before diving into the methods.
        """

        def rearrange_class_assignments(node: ast.ClassDef):
            b: list[ast.AST] = node.body
            x = []
            for i, item in enumerate(b):
                if isinstance(item, ast.Assign):
                    x.insert(0, item)
                else:
                    x.append(item)
            setattr(node, 'body', x)
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                rearrange_class_assignments(node)
        self._rewriteSource()

    def alphabetize_methods(self):
        pass

    def _name_is_constant(self, name: ast.Name):
        return name.id.isupper()

    def _assign_is_constant(self, node: ast.Assign):
        return len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0])

    def remove_duplicate_functions(self):
        functions = []
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                string = ast.unparse(node)
                if string in functions:
                    self.tree.body.remove(node)
                functions.append(string)
        self._rewriteSource()

    def remove_duplicate_constants(self):
        """
        Removes any duplicate constant assignments from the source code.

        This method goes through the Abstract Syntax Tree (AST) and finds all constant assignments.
        It then checks if any of the constant assignments are duplicates.  If a duplicate is found,
        it is removed from the AST.  Finally, the source code string is updated from the modified AST.
        """
        constants = []
        body = self.tree.body.copy()
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and self._assign_is_constant(node) == True:
                string = ast.unparse(node)
                if string in constants:
                    body.remove(node)
                constants.append(string)
        self._body(body)
        self._rewriteSource()

    def format_with_autopep8(self):
        self.source = autopep8.fix_code(self.source)
        self.tree = ast.parse(self.source)
        self._format = 'autopep8'

    def alphabetize_imports(self):
        """
        alphabetize imports
        """
        firstindex = len(self.tree.body)
        for index, node in enumerate(self.tree.body):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                firstindex = index
                break
        imps = []
        for index, node in enumerate(self.tree.body[firstindex:]):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imps.append(ast.unparse(node))
                self.tree.body.remove(node)
        imps.sort()
        imps = [ast.parse(imp).body[0] for imp in imps]
        self.tree.body = imps + self.tree.body
        self._rewriteSource()

    def group_constants_together(self):
        """
        Groups all constant assignments (assignments to uppercase names) together

        This function will go through the Abstract Syntax Tree (AST) and find the first constant assignment. 
        It will then group all following constant assignments together, and move them all to the front of the file.
        """
        first_constant = len(self.tree.body)
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
                    first_constant = self.tree.body.index(node)
                    break
        cons = []
        for node in self.tree.body[first_constant:]:
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
                    cons.append(node)
                else:
                    break
            else:
                break
        self.tree.body = self.tree.body[:first_constant] + \
            cons + self.tree.body[first_constant:]
        self._rewriteSource()

    def put_nameismain_last(self):
        """Moves any if __name__ == "__main__": blocks to the end of the file.

        This is done by iterating over the body of the current AST, and 
        storing any non-if __name__ == "__main__": blocks in a list. 
        If any if __name__ == "__main__": blocks are found, they are stored 
        in a separate list. If no if __name__ == "__main__": blocks are found, 
        this method does nothing. If any are found, they are appended to the 
        end of the list of non-if __name__ == "__main__": blocks and the body 
        of the current AST is updated to this new list. The source code string 
        is then updated from the current AST.
        """
        ifname_body = []
        keeping = []
        has_main = False
        for node in self.tree.body:
            if isinstance(node, ast.If) and is_if_name_is_main(node):
                ifname_body.extend(node.body)
                has_main = True
            else:
                keeping.append(node)
        if has_main == False:
            return
        ifn = ast.If(test=ast.Compare(left=ast.Name(id='__name__', ctx=ast.Load()), ops=[
                     ast.Eq()], comparators=[ast.Constant('__main__')]), body=ifname_body)
        keeping.append(ifn)
        self._body(keeping)
        self._rewriteSource()

    def main_function_is_last(self):
        mainfunc = None
        lastindex = 0
        c = 0
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                lastindex = c
                if node.name == 'main':
                    mainfunc = node
            c += 1
        if mainfunc != None:
            self.tree.body.remove(mainfunc)
            self.tree.body.insert(lastindex, mainfunc)
            self._rewriteSource()

    def private_methods_first(self):
        """
        for classes, write private methods first before public methods
        """
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        if subnode.name.startswith('_') == False:
                            node.body.remove(subnode)
                            node.body.insert(-1, subnode)
        self._rewriteSource()

    def remove_relative_imports(self):
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.level != 0:
                    self.tree.body.remove(node)
        self._rewriteSource()

    def remove_duplicate_classes(self):
        classes = []
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                if node in classes:
                    self.tree.body.remove(node)
                classes.append(node)
        self._rewriteSource()


def find_split_position(line: str, max_length: int) -> int:
    """
    Finds the rightmost position where the line can be split
    without breaking Python syntax and adhering to a maximum line length.

    Args:
        line (str): A line of Python source code.
        max_length (int): The maximum allowed line length.

    Returns:
        int: The index to split the line, or -1 if no valid split is found.
    """
    if len(line) <= max_length:
        return -1
    tokens = list(tokenize.generate_tokens(StringIO(line).readline))
    split_candidates = []
    for i, token in enumerate(tokens):
        tok_type, tok_string, start, end, _ = token
        if tok_type == tokenize.OP and tok_string in {',', '+', '-', '*', '/', '%', '(', '[', '{'}:
            split_candidates.append(start[1])
    for token in tokens:
        tok_type, _, start, _, _ = token
        if tok_type in {tokenize.STRING, tokenize.COMMENT}:
            col = start[1]
            split_candidates = [pos for pos in split_candidates if pos < col]
    split_candidates = [pos for pos in split_candidates if pos <=
                        max_length and len(line) - pos <= max_length]
    return max(split_candidates) if split_candidates else -1


@dataclass
class Problem:
    file: str
    line: int
    char: int
    code: str
    message: str

    def fromLine(line: str) -> Optional['Problem']:
        data = line.split(':')
        if len(data) > 3:
            return Problem(data[0].strip(), int(data[1].strip()), int(data[2].strip()), data[3].strip(), data[4].strip()[0:data[4].index('(') - 1], data[4].strip().split('(')[1].split(')')[0].strip())
        else:
            return None
    error: str


class ProblemFixer:

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.problems: list[Problem] = []

    def setsrc(self, source: str):
        self.source = source

    def FIX_C0303(self, problem: Problem):
        line_index = problem.line
        lines = self.source.split('\n')
        if line_index < 0 or line_index >= len(lines):
            print('Line index out of range')
            return
        lines[line_index] = lines[line_index].rstrip()
        self.source = '\n'.join(lines)
        print('Fixed trailing whitespace on line ', line_index)
        lines[line_index] = lines[line_index].rstrip()
        self.source = '\n'.join(lines)
        print('Fixed trailing whitespace on line ', line_index)

    def _indentation(self, line: str):
        return len(line) - len(line.lstrip())

    def FIX_C0301(self, problem: Problem):
        """
        Fixes a line that is too long.
        """
        maxlinelength = 78
        line_index = problem.line
        lines = self.source.split('\n')
        lines[line_index] = lines[line_index].rstrip()
        indent = ' ' * self._indentation(lines[line_index])
        nextline = indent + lines[line_index][maxlinelength:]
        thisline = lines[line_index][:maxlinelength] + '\\'
        lines[line_index] = thisline
        lines.insert(line_index + 1, nextline)
        self.source = '\n'.join(lines)
        print('Fixed line length on line ', line_index)
        nextline = indent + lines[line_index][maxlinelength:]
        thisline = lines[line_index][:maxlinelength] + '\\'
        self.source = '\n'.join(lines)
        print('Fixed line length on line ', line_index)

    def FIX_R1705(self, problem: Problem) -> None:
        """
        Fixes a Unnecessary `elif` after `return`, remove the leading `el` from `elif`.
        """
        try:
            line_number = problem.line
            lines = self.source.splitlines()
            if line_number is None or line_number >= len(lines):
                return
            line = lines[line_number]
            if 'elif' not in line:
                return
            lines[line_number] = line.replace('elif', 'if')
            self.source = '\n'.join(lines)
        except Exception as e:
            print(f'An error occurred while fixing unnecessary elif: {e}')

    def FIX_W0621(self, problem: Problem) -> None:
        """
        Fixes a Redefining name X from outer scope.

        The W0621 error occurs when a variable is redefined in a nested scope.
        This method fixes it by changing the line of code that defines the variable.

        Args:
            problem (Problem): The problem object from the linter that contains the
                error message and the line number of the error.
        """
        try:
            line_number = problem.line
            if line_number is None:
                print('Line number is None')
                return
            lines = self.source.splitlines()
            if line_number >= len(lines) or line_number < 0:
                print('Line index out of range')
                return
            line = lines[line_number]
            if 'def' not in line:
                print('No defining found on line ', line_number)
                return
            lines[line_number] = line.replace('def', 'async def')
            self.source = '\n'.join(lines)
        except IndexError:
            print(f'Line index out of range: {line_number}')
        except Exception as e:
            print(f'An error occurred while fixing redefining name: {e}')

    def fix_problem(self):
        """
        Fixes the problems in the source code.

        This method iterates over the list of problems found by the linter and
        calls the corresponding FIX_<code> method to fix the problem.

        If no FIX_<code> method exists for the given problem, the problem is
        skipped.

        Returns:
            None
        """
        self.lint()
        if len(self.problems) == 0:
            print('No problems to fix')
            return
        for problem in self.problems:
            method = 'FIX_' + problem.code.upper()
            if not hasattr(self, method):
                continue
            else:
                method = getattr(self, method)
                method(problem)

    def FIX_C0304(self, problem: Problem):
        if not self.source.endswith('\n'):
            self.source += '\n'
            print('Fixed missing newline at end of file')
        else:
            print('No missing newline at end of file')

    def FIX_W0621(self, problem: Problem):
        """
        Fixes a Redefining name X from outer scope.

        This happens when a variable is defined in an outer scope and then
        redefined in an inner scope. For example:

        x = 1
        def f():
            x = 2

        This is legal Python but can lead to confusing code. This function
        fixes the issue by removing the redefinition, so that the variable
        is no longer redefined in the inner scope.
        """
        try:
            line_index = problem.line
            lines = self.source.split('\n')
            if line_index >= len(lines) or line_index < 0:
                print('Line index out of range')
                return
            line = lines[line_index]
            if not line.startswith('    ') and (not line.startswith('\t')):
                print('No redefinition found on line ', line_index)
                return
            if '"""' in line or "'''" in line:
                print('No redefinition found on line ', line_index)
                return
            lines[line_index] = ''
            self.source = '\n'.join(lines)
            print('Fixed redefining name on line ', line_index)
        except Exception as e:
            print(f'An error occurred while fixing redefining name: {e}')

    def FIX_C0115(self, problem: Problem) -> None:
        """
        Fixes a missing function or method docstring.
        """
        try:
            line_number = problem.line
            if line_number is None:
                print('Line number is None')
                return
            lines = self.source.splitlines()
            if line_number >= len(lines) or line_number < 0:
                print('Line index out of range')
                return
            line = lines[line_number]
            if '"""' in line:
                print('Docstring already present on line', line_number)
                return
            docstring = '"""Add docstring here"""'
            lines.insert(line_number, docstring)
            self.source = '\n'.join(lines)
            print('Added docstring on line', line_number)
        except Exception as e:
            print(f'An error occurred while fixing missing docstring: {e}')

    def FIX_C0103(self, problem: Problem) -> None:
        """
        Fixes a Constant name X doesn't conform to UPPER_CASE naming style.

        Args:
            problem (dict): A dictionary containing the problem information.

        Returns:
            None
        """
        try:
            line_index = problem.line
            if line_index is None:
                print('Problem does not contain a line number')
                return
            lines = self.source.split('\n')
            if line_index >= len(lines) or line_index < 0:
                print('Line index out of range')
                return
            line = lines[line_index]
            if '=' not in line:
                print('No assignment found on line ', line_index)
                return
            left, right = line.split('=', 1)
            left = left.strip()
            left = left.upper()
            lines[line_index] = f'{left}={right}'
            self.source = '\n'.join(lines)
            print('Fixed constant name on line ', line_index)
        except Exception as e:
            print('Error fixing constant name: ', e)

    def lint(self) -> None:
        """
        Runs pylint on the source code and saves the result to self.problems.

        The problems are saved as a list of Problem objects, which contain the following
        information:

            file: str
            line: int
            char: int
            code: str
            message: str
            error: str

        The problems are only saved if no exception occurs while running pylint.
        If an exception occurs, the problems are not saved and the exception is propagated.
        """
        tempfile = 'temp_file.py'
        with open(tempfile, 'w', encoding='utf-8') as f:
            f.write(self.source)
        try:
            result = subprocess.run(
                ['pylint', tempfile], capture_output=True, text=True)
            output = result.stdout
            problems = []
            lines = output.split('\n')
            for line in lines:
                prob = Problem.fromLine(line)
                if prob is not None:
                    problems.append(prob)
        except Exception as e:
            return
        finally:
            self.problems = problems

    def fix_all_problems(self):
        """
        Fixes all problems in the source code.

        This method iterates over the list of problems found by the linter and
        calls the corresponding FIX_<code> method to fix the problem.

        If no FIX_<code> method exists for the given problem, the problem is
        skipped.

        Returns:
            None
        """
        self.lint()
        if len(self.problems) == 0:
            print('No problems to fix')
            return
        for problem in self.problems:
            method = 'FIX_' + problem.code.upper()
            if not hasattr(self, method):
                continue
            else:
                method = getattr(self, method)
                method(problem)

    def FIX_R1728(self, problem: Problem) -> None:
        """
        Fixes Consider using a generator inst
        """
        try:
            line_number = problem.line
            lines = self.source.splitlines()
            if line_number is None or line_number >= len(lines):
                return
            line = lines[line_number]
            if 'for' not in line:
                return
            lines[line_number] = line.replace('for', 'if')
            self.source = '\n'.join(lines)
        except Exception as e:
            print(f'An error occurred while fixing unnecessary elif: {e}')


@dataclass
class Problem:
    file: str
    line: int
    char: int
    code: str
    message: str
    error: str

    def __str__(self):
        return f'{self.file}:{self.line}:{self.char}:{self.code}:{self.message}:{self.error}'

    @classmethod
    def fromLine(cls, line: str):
        data = line.split(':')
        if len(data) > 3:
            d = {'file': data[0].strip(), 'line': int(data[1].strip()), 'char': int(data[2].strip()), 'code': data[3].strip(
            ), 'message': data[4].strip()[0:data[4].index('(') - 1], 'error': data[4].strip().split('(')[1].split(')')[0].strip()}
            return cls(**d)
        else:
            return None

    def __repr__(self):
        return f'{self.file}:{self.line}:{self.char}:{self.code}:{self.message}:{self.error}'


class SourceLinter:

    def __init__(self, source: str):
        """
        Initialize the SourceLinter with the given source code.

        Args:
            source (str): The source code to be linted.
        """
        self.source = source

    def lint(self) -> str:
        """
        Runs pylint on the source code and returns the output.

        The source code is saved to a temporary file and then pylint is run as a subprocess.
        The output from pylint is captured and returned as a string.

        If an exception occurs while running pylint, the function returns a string containing the exception message.

        Returns:
            str: The output from pylint or an error message if an exception occurred.
        """
        temp_file = 'temp_file.py'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(self.source)
        try:
            result = subprocess.run(
                ['pylint', temp_file], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            return f'Linter failed: {e}'

    def get_problems(self) -> list[Problem]:
        """
        Executes the linter on the source code and returns a list of linting issues.

        The source code is first saved to a temporary file, and the linter is executed
        as a subprocess. The output is parsed to extract the issues, which are returned
        as a list of Problem objects.

        Returns:
            list[Problem]: A list of Problem objects representing the linting issues.
        """
        temp_file = 'temp_file.py'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(self.source)
        try:
            result = subprocess.run(
                ['pylint', temp_file], capture_output=True, text=True)
            output = result.stdout
        except Exception as e:
            return f'Linter failed: {e}'
        problems = []
        lines = output.split('\n')
        for line in lines:
            data = line.split(':')
            if len(data) > 3:
                problems.append(Problem.fromLine(line))
        return problems

    def parse_pylint_output(self, output: str) -> list[Problem]:
        """
        Parses the output from pylint and returns a string representation of the issues.

        Args:
            output (str): The output from pylint.

        Returns:
            str: A string representation of the issues.
        """
        problems = []
        lines = output.split('\n')
        for line in lines:
            data = line.split(':')
            if len(data) > 3:
                problems.append(Problem.fromLine(line))
        return problems

    def lint_summary(self, linter='pylint') -> str:
        """
        Provides a concise summary of linting results.

        Args:
            linter (str): The linter to use ('pylint' or 'flake8').

        Returns:
            str: A brief summary of linting results, including issue count.
        """
        output = self.lint(linter)
        if linter == 'pylint':
            output = self.parse_pylint_output(output)
            for issue in output:
                print(issue)

    def write(self):
        """
        Writes the source code to a temporary file.

        The temporary file is named "temp_file.py" and is placed in the current working directory.
        The file is written in UTF-8 encoding.
        """
        with open('temp_file.py', 'w', encoding='utf-8') as f:
            f.write(self.source)


def red(text):
    return f'\x1b[91m{text}\x1b[0m'


def background_red(text):
    return f'\x1b[41m{text}\x1b[0m'


def redline(text):
    print(f'\x1b[91m{text}\x1b[0m')


def green(text):
    return f'\x1b[92m{text}\x1b[0m'


def greenline(text):
    print(f'\x1b[92m{text}\x1b[0m')


def blue(text):
    return f'\x1b[94m{text}\x1b[0m'


def blueline(text):
    print(f'\x1b[94m{text}\x1b[0m')


def dim(text):
    return f'{DIMBLACK}{text}\x1b[0m'


class SourceReader:

    def __init__(self, source: str):
        self.source = source
        self.problems: list[Problem] = []

    def react(self, problems: list[Problem]):
        """
        Reacts to linting problems by printing the source code with linting problems
        highlighted in red and good lines in green. The output is formatted so that
        the leftmost column is the line number and the next column is the source code
        line. The rightmost column is reserved for the linting problem message, if
        applicable.

        Args:
            problems (list[dict]): A list of dictionaries representing the linting issues.

        Returns:
            None
        """
        problem_lines = [problem.line for problem in problems]
        with open(problems[0].file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        max_length = len(str(len(lines)))
        max_line_length = max([len(line) for line in lines]) + 5
        max_terminal_width = shutil.get_terminal_size().columns
        additional_text = ''
        for index, line in enumerate(lines):
            line = line.rstrip()
            index = index + 1
            if index in problem_lines:
                problem = [p for p in problems if p.line == index][0]
                icon = self.get_icon(problem)
                if problem.char == 0:
                    line = f'{icon + WALL}' + \
                        str(index).ljust(max_length) + WALL + red(line)
                    line = line.ljust(max_line_length) + WALL + \
                        blue(problem.code) + ' ' + problem.message
                else:
                    line = f'{icon + WALL}' + str(index).ljust(
                        max_length) + WALL + line[:problem.char] + background_red(line[problem.char:])
                    line = line.ljust(max_line_length) + WALL + \
                        blue(problem.code) + ' ' + problem.message
            else:
                icon = GOOD
                line = f'{icon + WALL}' + \
                    str(index).ljust(max_length) + WALL + dim(line)
                line = line.ljust(max_line_length) + WALL
            if len(line) > max_terminal_width:
                print(line[:max_terminal_width])
                additional_text = '     ' + line[max_terminal_width:]
            else:
                if additional_text != '':
                    line = line + additional_text
                    additional_text = ''
                print(line)

    def get_icon(self, problem: Problem):
        if isinstance(problem, type(None)):
            return GOOD
        if problem.code.startswith('E'):
            return FATAL
        elif problem.code.startswith('W'):
            return WARN
        elif problem.code.startswith('C'):
            return CONV
        elif problem.code.startswith('R'):
            return RECM
        else:
            return GOOD


def simple_enum(etype=Enum, *, boundary=None, use_args=None):
    """
    Class decorator that converts a normal class into an :class:`Enum`.  No
    safety checks are done, and some advanced behavior (such as
    :func:`__init_subclass__`) is not available.  Enum creation can be faster
    using :func:`_simple_enum`.

        >>> from enum import Enum, _simple_enum
        >>> @_simple_enum(Enum)
        ... class Color:
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> Color
        <enum 'Color'>
    """

    def convert_class(cls):
        nonlocal use_args
        cls_name = cls.__name__
        if use_args is None:
            use_args = etype._use_args_
        __new__ = cls.__dict__.get('__new__')
        if __new__ is not None:
            new_member = __new__.__func__
        else:
            new_member = etype._member_type_.__new__
        attrs = {}
        body = {}
        if __new__ is not None:
            body['__new_member__'] = new_member
        body['_new_member_'] = new_member
        body['_use_args_'] = use_args
        body['_generate_next_value_'] = gnv = etype._generate_next_value_
        body['_member_names_'] = member_names = []
        body['_member_map_'] = member_map = {}
        body['_value2member_map_'] = value2member_map = {}
        body['_unhashable_values_'] = unhashable_values = []
        body['_unhashable_values_map_'] = {}
        body['_member_type_'] = member_type = etype._member_type_
        body['_value_repr_'] = etype._value_repr_
        if issubclass(etype, Flag):
            body['_boundary_'] = boundary or etype._boundary_
            body['_flag_mask_'] = None
            body['_all_bits_'] = None
            body['_singles_mask_'] = None
            body['_inverted_'] = None
            body['__or__'] = Flag.__or__
            body['__xor__'] = Flag.__xor__
            body['__and__'] = Flag.__and__
            body['__ror__'] = Flag.__ror__
            body['__rxor__'] = Flag.__rxor__
            body['__rand__'] = Flag.__rand__
            body['__invert__'] = Flag.__invert__
        for name, obj in cls.__dict__.items():
            if name in ('__dict__', '__weakref__'):
                continue
            if _is_dunder(name) or _is_private(cls_name, name) or _is_sunder(name) or _is_descriptor(obj):
                body[name] = obj
            else:
                attrs[name] = obj
        if cls.__dict__.get('__doc__') is None:
            body['__doc__'] = 'An enumeration.'
        enum_class = type(cls_name, (etype,), body,
                          boundary=boundary, _simple=True)
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name not in body:
                enum_method = getattr(etype, name)
                found_method = getattr(enum_class, name)
                object_method = getattr(object, name)
                data_type_method = getattr(member_type, name)
                if found_method in (data_type_method, object_method):
                    setattr(enum_class, name, enum_method)
        gnv_last_values = []
        if issubclass(enum_class, Flag):
            single_bits = multi_bits = 0
            for name, value in attrs.items():
                if isinstance(value, auto) and auto.value is _auto_null:
                    value = gnv(name, 1, len(member_names), gnv_last_values)
                if use_args:
                    if not isinstance(value, tuple):
                        value = (value,)
                    member = new_member(enum_class, *value)
                    value = value[0]
                else:
                    member = new_member(enum_class)
                if __new__ is None:
                    member._value_ = value
                try:
                    contained = value2member_map.get(member._value_)
                except TypeError:
                    contained = None
                    if member._value_ in unhashable_values:
                        for m in enum_class:
                            if m._value_ == member._value_:
                                contained = m
                                break
                if contained is not None:
                    contained._add_alias_(name)
                else:
                    member._name_ = name
                    member.__objclass__ = enum_class
                    member.__init__(value)
                    member._sort_order_ = len(member_names)
                    if name not in ('name', 'value'):
                        setattr(enum_class, name, member)
                        member_map[name] = member
                    else:
                        enum_class._add_member_(name, member)
                    value2member_map[value] = member
                    if _is_single_bit(value):
                        member_names.append(name)
                        single_bits |= value
                    else:
                        multi_bits |= value
                    gnv_last_values.append(value)
            enum_class._flag_mask_ = single_bits | multi_bits
            enum_class._singles_mask_ = single_bits
            enum_class._all_bits_ = 2 ** (single_bits |
                                          multi_bits).bit_length() - 1
            member_list = [m._value_ for m in enum_class]
            if member_list != sorted(member_list):
                enum_class._iter_member_ = enum_class._iter_member_by_def_
        else:
            for name, value in attrs.items():
                if isinstance(value, auto):
                    if value.value is _auto_null:
                        value.value = gnv(name, 1, len(
                            member_names), gnv_last_values)
                    value = value.value
                if use_args:
                    if not isinstance(value, tuple):
                        value = (value,)
                    member = new_member(enum_class, *value)
                    value = value[0]
                else:
                    member = new_member(enum_class)
                if __new__ is None:
                    member._value_ = value
                try:
                    contained = value2member_map.get(member._value_)
                except TypeError:
                    contained = None
                    if member._value_ in unhashable_values:
                        for m in enum_class:
                            if m._value_ == member._value_:
                                contained = m
                                break
                if contained is not None:
                    contained._add_alias_(name)
                else:
                    member._name_ = name
                    member.__objclass__ = enum_class
                    member.__init__(value)
                    member._sort_order_ = len(member_names)
                    if name not in ('name', 'value'):
                        setattr(enum_class, name, member)
                        member_map[name] = member
                    else:
                        enum_class._add_member_(name, member)
                    member_names.append(name)
                    gnv_last_values.append(value)
                    try:
                        enum_class._value2member_map_.setdefault(value, member)
                    except TypeError:
                        enum_class._unhashable_values_.append(value)
                        enum_class._unhashable_values_map_.setdefault(
                            name, []).append(value)
        if '__new__' in body:
            enum_class.__new_member__ = enum_class.__new__
        enum_class.__new__ = Enum.__new__
        return enum_class
    return convert_class


@simple_enum(IntEnum)
class Precedence:
    ATOM = auto()
    AWAIT = auto()
    POWER = auto()
    FACTOR = auto()
    TERM = auto()
    ARITH = auto()
    SHIFT = auto()
    BAND = auto()
    BXOR = auto()
    BOR = EXPR
    EXPR = auto()
    CMP = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    TEST = auto()
    YIELD = auto()
    TUPLE = auto()
    NAMED_EXPR = auto()

    def next(self):
        try:
            return self.__class__(self + 1)
        except ValueError:
            return self
    'Precedence table that originated from python grammar.'


class Unparser(NodeVisitor):
    boolop_precedence = {'and': Precedence.AND, 'or': Precedence.OR}
    boolops = {'And': 'and', 'Or': 'or'}
    cmpops = {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=', 'Gt': '>',
              'GtE': '>=', 'Is': 'is', 'IsNot': 'is not', 'In': 'in', 'NotIn': 'not in'}
    binop_rassoc = frozenset(('**',))
    binop_precedence = {'+': Precedence.ARITH, '-': Precedence.ARITH, '*': Precedence.TERM, '@': Precedence.TERM, '/': Precedence.TERM, '%': Precedence.TERM,
                        '<<': Precedence.SHIFT, '>>': Precedence.SHIFT, '|': Precedence.BOR, '^': Precedence.BXOR, '&': Precedence.BAND, '//': Precedence.TERM, '**': Precedence.POWER}
    binop = {'Add': '+', 'Sub': '-', 'Mult': '*', 'MatMult': '@', 'Div': '/', 'Mod': '%', 'LShift': '<<',
             'RShift': '>>', 'BitOr': '|', 'BitXor': '^', 'BitAnd': '&', 'FloorDiv': '//', 'Pow': '**'}
    unop_precedence = {'not': Precedence.NOT, '~': Precedence.FACTOR,
                       '+': Precedence.FACTOR, '-': Precedence.FACTOR}
    unop = {'Invert': '~', 'Not': 'not', 'UAdd': '+', 'USub': '-'}
    'Methods in this class recursively traverse an AST and\n    output source code for the abstract syntax; original formatting\n    is disregarded.'

    def __init__(self):
        self._source = []
        self._precedences = {}
        self._type_ignores = {}
        self._indent = 0
        self._in_try_star = False

    def items_view(self, traverser, items):
        """Traverse and separate the given *items* with a comma and append it to
        the buffer. If *items* is a single item sequence, a trailing comma
        will be added."""
        if len(items) == 1:
            traverser(items[0])
            self.write(',')
        else:
            self.interleave(lambda: self.write(', '), traverser, items)

    def fill(self, text=''):
        """Indent a piece of text and append it, according to the current
        indentation level"""
        self.maybe_newline()
        self.write('    ' * self._indent + text)

    @contextmanager
    def buffered(self, buffer=None):
        if buffer is None:
            buffer = []
        original_source = self._source
        self._source = buffer
        yield buffer
        self._source = original_source

    @contextmanager
    def delimit(self, start, end):
        """A context manager for preparing the source for expressions. It adds
        *start* to the buffer and enters, after exit it adds *end*."""
        self.write(start)
        yield
        self.write(end)

    def require_parens(self, precedence, node):
        """Shortcut to adding precedence related parens"""
        return self.delimit_if('(', ')', self.get_precedence(node) > precedence)

    def set_precedence(self, precedence, *nodes):
        for node in nodes:
            self._precedences[node] = precedence

    def get_type_comment(self, node):
        comment = self._type_ignores.get(node.lineno) or node.type_comment
        if comment is not None:
            return f' # type: {comment}'

    def visit(self, node):
        """Outputs a source code string that, if converted back to an ast
        (using ast.parse) will generate an AST equivalent to *node*"""
        self._source = []
        self.traverse(node)
        return ''.join(self._source)

    def _write_docstring_and_traverse_body(self, node):
        if (docstring := self.get_raw_docstring(node)):
            self._write_docstring(docstring)
            self.traverse(node.body[1:])
        else:
            self.traverse(node.body)

    def visit_FunctionType(self, node):
        with self.delimit('(', ')'):
            self.interleave(lambda: self.write(', '),
                            self.traverse, node.argtypes)
        self.write(' -> ')
        self.traverse(node.returns)

    def visit_NamedExpr(self, node):
        with self.require_parens(Precedence.NAMED_EXPR, node):
            self.set_precedence(Precedence.ATOM, node.target, node.value)
            self.traverse(node.target)
            self.write(' := ')
            self.traverse(node.value)

    def visit_ImportFrom(self, node):
        self.fill('from ')
        self.write('.' * (node.level or 0))
        if node.module:
            self.write(node.module)
        self.write(' import ')
        self.interleave(lambda: self.write(', '), self.traverse, node.names)

    def visit_AugAssign(self, node):
        self.fill()
        self.traverse(node.target)
        self.write(' ' + self.binop[node.op.__class__.__name__] + '= ')
        self.traverse(node.value)

    def visit_Return(self, node):
        self.fill('return')
        if node.value:
            self.write(' ')
            self.traverse(node.value)

    def visit_Break(self, node):
        self.fill('break')

    def visit_Delete(self, node):
        self.fill('del ')
        self.interleave(lambda: self.write(', '), self.traverse, node.targets)

    def visit_Global(self, node):
        self.fill('global ')
        self.interleave(lambda: self.write(', '), self.write, node.names)

    def visit_Await(self, node):
        with self.require_parens(Precedence.AWAIT, node):
            self.write('await')
            if node.value:
                self.write(' ')
                self.set_precedence(Precedence.ATOM, node.value)
                self.traverse(node.value)

    def visit_YieldFrom(self, node):
        with self.require_parens(Precedence.YIELD, node):
            self.write('yield from ')
            if not node.value:
                raise ValueError(
                    "Node can't be used without a value attribute.")
            self.set_precedence(Precedence.ATOM, node.value)
            self.traverse(node.value)

    def do_visit_try(self, node):
        self.fill('try')
        with self.block():
            self.traverse(node.body)
        for ex in node.handlers:
            self.traverse(ex)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.traverse(node.orelse)
        if node.finalbody:
            self.fill('finally')
            with self.block():
                self.traverse(node.finalbody)

    def visit_TryStar(self, node):
        prev_in_try_star = self._in_try_star
        try:
            self._in_try_star = True
            self.do_visit_try(node)
        finally:
            self._in_try_star = prev_in_try_star

    def visit_ClassDef(self, node):
        self.maybe_newline()
        for deco in node.decorator_list:
            self.fill('@')
            self.traverse(deco)
        self.fill('class ' + node.name)
        if hasattr(node, 'type_params'):
            self._type_params_helper(node.type_params)
        with self.delimit_if('(', ')', condition=node.bases or node.keywords):
            comma = False
            for e in node.bases:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.traverse(e)
        with self.block():
            self._write_docstring_and_traverse_body(node)

    def visit_AsyncFunctionDef(self, node):
        self._function_helper(node, 'async def')

    def _function_helper(self, node, fill_suffix):
        self.maybe_newline()
        for deco in node.decorator_list:
            self.fill('@')
            self.traverse(deco)
        def_str = fill_suffix + ' ' + node.name
        self.fill(def_str)
        if hasattr(node, 'type_params'):
            self._type_params_helper(node.type_params)
        with self.delimit('(', ')'):
            self.traverse(node.args)
        if node.returns:
            self.write(' -> ')
            self.traverse(node.returns)
        with self.block(extra=self.get_type_comment(node)):
            self._write_docstring_and_traverse_body(node)

    def _type_params_helper(self, type_params):
        if type_params is not None and len(type_params) > 0:
            with self.delimit('[', ']'):
                self.interleave(lambda: self.write(', '),
                                self.traverse, type_params)

    def visit_TypeVarTuple(self, node):
        self.write('*' + node.name)
        if node.default_value:
            self.write(' = ')
            self.traverse(node.default_value)

    def visit_TypeAlias(self, node):
        self.fill('type ')
        self.traverse(node.name)
        self._type_params_helper(node.type_params)
        self.write(' = ')
        self.traverse(node.value)

    def visit_AsyncFor(self, node):
        self._for_helper('async for ', node)

    def _for_helper(self, fill, node):
        self.fill(fill)
        self.set_precedence(Precedence.TUPLE, node.target)
        self.traverse(node.target)
        self.write(' in ')
        self.traverse(node.iter)
        with self.block(extra=self.get_type_comment(node)):
            self.traverse(node.body)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.traverse(node.orelse)

    def visit_While(self, node):
        self.fill('while ')
        self.traverse(node.test)
        with self.block():
            self.traverse(node.body)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.traverse(node.orelse)

    def visit_AsyncWith(self, node):
        self.fill('async with ')
        self.interleave(lambda: self.write(', '), self.traverse, node.items)
        with self.block(extra=self.get_type_comment(node)):
            self.traverse(node.body)

    def _str_literal_helper(self, string, *, quote_types=_ALL_QUOTES, escape_special_whitespace=False):
        """Helper for writing string literals, minimizing escapes.
        Returns the tuple (string literal to write, possible quote types).
        """

        def escape_char(c):
            if not escape_special_whitespace and c in '\n\t':
                return c
            if c == '\\' or not c.isprintable():
                return c.encode('unicode_escape').decode('ascii')
            return c
        escaped_string = ''.join(map(escape_char, string))
        possible_quotes = quote_types
        if '\n' in escaped_string:
            possible_quotes = [
                q for q in possible_quotes if q in _MULTI_QUOTES]
        possible_quotes = [
            q for q in possible_quotes if q not in escaped_string]
        if not possible_quotes:
            string = repr(string)
            quote = next((q for q in quote_types if string[0] in q), string[0])
            return (string[1:-1], [quote])
        if escaped_string:
            possible_quotes.sort(key=lambda q: q[0] == escaped_string[-1])
            if possible_quotes[0][0] == escaped_string[-1]:
                assert len(possible_quotes[0]) == 3
                escaped_string = escaped_string[:-
                                                1] + '\\' + escaped_string[-1]
        return (escaped_string, possible_quotes)

    def _write_str_avoiding_backslashes(self, string, *, quote_types=_ALL_QUOTES):
        """Write string literal value with a best effort attempt to avoid backslashes."""
        string, quote_types = self._str_literal_helper(
            string, quote_types=quote_types)
        quote_type = quote_types[0]
        self.write(f'{quote_type}{string}{quote_type}')

    def _write_fstring_inner(self, node, is_format_spec=False):
        if isinstance(node, JoinedStr):
            for value in node.values:
                self._write_fstring_inner(value, is_format_spec=is_format_spec)
        elif isinstance(node, Constant) and isinstance(node.value, str):
            value = node.value.replace('{', '{{').replace('}', '}}')
            if is_format_spec:
                value = value.replace('\\', '\\\\')
                value = value.replace("'", "\\'")
                value = value.replace('"', '\\"')
                value = value.replace('\n', '\\n')
            self.write(value)
        elif isinstance(node, FormattedValue):
            self.visit_FormattedValue(node)
        else:
            raise ValueError(f'Unexpected node inside JoinedStr, {node!r}')

    def visit_Name(self, node):
        self.write(node.id)

    def _write_docstring(self, node):
        self.fill()
        if node.kind == 'u':
            self.write('u')
        self._write_str_avoiding_backslashes(
            node.value, quote_types=_MULTI_QUOTES)

    def _write_constant(self, value):
        if isinstance(value, (float, complex)):
            self.write(repr(value).replace('inf', _INFSTR).replace(
                'nan', f'({_INFSTR}-{_INFSTR})'))
        else:
            self.write(repr(value))

    def visit_List(self, node):
        with self.delimit('[', ']'):
            self.interleave(lambda: self.write(', '), self.traverse, node.elts)

    def visit_GeneratorExp(self, node):
        with self.delimit('(', ')'):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_DictComp(self, node):
        with self.delimit('{', '}'):
            self.traverse(node.key)
            self.write(': ')
            self.traverse(node.value)
            for gen in node.generators:
                self.traverse(gen)

    def visit_IfExp(self, node):
        with self.require_parens(Precedence.TEST, node):
            self.set_precedence(Precedence.TEST.next(), node.body, node.test)
            self.traverse(node.body)
            self.write(' if ')
            self.traverse(node.test)
            self.write(' else ')
            self.set_precedence(Precedence.TEST, node.orelse)
            self.traverse(node.orelse)

    def visit_Dict(self, node):

        def write_key_value_pair(k, v):
            self.traverse(k)
            self.write(': ')
            self.traverse(v)

        def write_item(item):
            k, v = item
            if k is None:
                self.write('**')
                self.set_precedence(Precedence.EXPR, v)
                self.traverse(v)
            else:
                write_key_value_pair(k, v)
        with self.delimit('{', '}'):
            self.interleave(lambda: self.write(', '),
                            write_item, zip(node.keys, node.values))

    def visit_UnaryOp(self, node):
        operator = self.unop[node.op.__class__.__name__]
        operator_precedence = self.unop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            self.write(operator)
            if operator_precedence is not Precedence.FACTOR:
                self.write(' ')
            self.set_precedence(operator_precedence, node.operand)
            self.traverse(node.operand)

    def visit_Compare(self, node):
        with self.require_parens(Precedence.CMP, node):
            self.set_precedence(Precedence.CMP.next(),
                                node.left, *node.comparators)
            self.traverse(node.left)
            for o, e in zip(node.ops, node.comparators):
                self.write(' ' + self.cmpops[o.__class__.__name__] + ' ')
                self.traverse(e)

    def visit_Attribute(self, node):
        self.set_precedence(Precedence.ATOM, node.value)
        self.traverse(node.value)
        if isinstance(node.value, Constant) and isinstance(node.value.value, int):
            self.write(' ')
        self.write('.')
        self.write(node.attr)

    def visit_Subscript(self, node):

        def is_non_empty_tuple(slice_value):
            return isinstance(slice_value, Tuple) and slice_value.elts
        self.set_precedence(Precedence.ATOM, node.value)
        self.traverse(node.value)
        with self.delimit('[', ']'):
            if is_non_empty_tuple(node.slice):
                self.items_view(self.traverse, node.slice.elts)
            else:
                self.traverse(node.slice)

    def visit_Ellipsis(self, node):
        self.write('...')

    def visit_Match(self, node):
        self.fill('match ')
        self.traverse(node.subject)
        with self.block():
            for case in node.cases:
                self.traverse(case)

    def visit_arguments(self, node):
        first = True
        all_args = node.posonlyargs + node.args
        defaults = [None] * \
            (len(all_args) - len(node.defaults)) + node.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:
                first = False
            else:
                self.write(', ')
            self.traverse(a)
            if d:
                self.write('=')
                self.traverse(d)
            if index == len(node.posonlyargs):
                self.write(', /')
        if node.vararg or node.kwonlyargs:
            if first:
                first = False
            else:
                self.write(', ')
            self.write('*')
            if node.vararg:
                self.write(node.vararg.arg)
                if node.vararg.annotation:
                    self.write(': ')
                    self.traverse(node.vararg.annotation)
        if node.kwonlyargs:
            for a, d in zip(node.kwonlyargs, node.kw_defaults):
                self.write(', ')
                self.traverse(a)
                if d:
                    self.write('=')
                    self.traverse(d)
        if node.kwarg:
            if first:
                first = False
            else:
                self.write(', ')
            self.write('**' + node.kwarg.arg)
            if node.kwarg.annotation:
                self.write(': ')
                self.traverse(node.kwarg.annotation)

    def visit_Lambda(self, node):
        with self.require_parens(Precedence.TEST, node):
            self.write('lambda')
            with self.buffered() as buffer:
                self.traverse(node.args)
            if buffer:
                self.write(' ', *buffer)
            self.write(': ')
            self.set_precedence(Precedence.TEST, node.body)
            self.traverse(node.body)

    def visit_withitem(self, node):
        self.traverse(node.context_expr)
        if node.optional_vars:
            self.write(' as ')
            self.traverse(node.optional_vars)

    def visit_MatchValue(self, node):
        self.traverse(node.value)

    def visit_MatchSequence(self, node):
        with self.delimit('[', ']'):
            self.interleave(lambda: self.write(', '),
                            self.traverse, node.patterns)

    def visit_MatchMapping(self, node):

        def write_key_pattern_pair(pair):
            k, p = pair
            self.traverse(k)
            self.write(': ')
            self.traverse(p)
        with self.delimit('{', '}'):
            keys = node.keys
            self.interleave(lambda: self.write(', '), write_key_pattern_pair, zip(
                keys, node.patterns, strict=True))
            rest = node.rest
            if rest is not None:
                if keys:
                    self.write(', ')
                self.write(f'**{rest}')

    def visit_MatchAs(self, node):
        name = node.name
        pattern = node.pattern
        if name is None:
            self.write('_')
        elif pattern is None:
            self.write(node.name)
        else:
            with self.require_parens(Precedence.TEST, node):
                self.set_precedence(Precedence.BOR, node.pattern)
                self.traverse(node.pattern)
                self.write(f' as {node.name}')

    def maybe_newline(self):
        """Adds a newline if it isn't the start of generated source"""
        if self._source:
            self.write('\n')

    @contextmanager
    def block(self, *, extra=None):
        """A context manager for preparing the source for blocks. It adds
        the character':', increases the indentation on enter and decreases
        the indentation on exit. If *extra* is given, it will be directly
        appended after the colon character.
        """
        self.write(':')
        if extra:
            self.write(extra)
        self._indent += 1
        yield
        self._indent -= 1

    def get_precedence(self, node):
        return self._precedences.get(node, Precedence.TEST)

    def traverse(self, node):
        if isinstance(node, list):
            for item in node:
                self.traverse(item)
        else:
            super().visit(node)

    def visit_Expr(self, node):
        self.fill()
        self.set_precedence(Precedence.YIELD, node.value)
        self.traverse(node.value)

    def visit_Assign(self, node):
        self.fill()
        for target in node.targets:
            self.set_precedence(Precedence.TUPLE, target)
            self.traverse(target)
            self.write(' = ')
        self.traverse(node.value)
        if (type_comment := self.get_type_comment(node)):
            self.write(type_comment)

    def visit_Pass(self, node):
        self.fill('pass')

    def visit_Assert(self, node):
        self.fill('assert ')
        self.traverse(node.test)
        if node.msg:
            self.write(', ')
            self.traverse(node.msg)

    def visit_Yield(self, node):
        with self.require_parens(Precedence.YIELD, node):
            self.write('yield')
            if node.value:
                self.write(' ')
                self.set_precedence(Precedence.ATOM, node.value)
                self.traverse(node.value)

    def visit_Try(self, node):
        prev_in_try_star = self._in_try_star
        try:
            self._in_try_star = False
            self.do_visit_try(node)
        finally:
            self._in_try_star = prev_in_try_star

    def visit_FunctionDef(self, node):
        self._function_helper(node, 'def')

    def visit_ParamSpec(self, node):
        self.write('**' + node.name)
        if node.default_value:
            self.write(' = ')
            self.traverse(node.default_value)

    def visit_If(self, node):
        self.fill('if ')
        self.traverse(node.test)
        with self.block():
            self.traverse(node.body)
        while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], If):
            node = node.orelse[0]
            self.fill('elif ')
            self.traverse(node.test)
            with self.block():
                self.traverse(node.body)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.traverse(node.orelse)

    def visit_JoinedStr(self, node):
        self.write('f')
        fstring_parts = []
        for value in node.values:
            with self.buffered() as buffer:
                self._write_fstring_inner(value)
            fstring_parts.append(
                (''.join(buffer), isinstance(value, Constant)))
        new_fstring_parts = []
        quote_types = list(_ALL_QUOTES)
        fallback_to_repr = False
        for value, is_constant in fstring_parts:
            if is_constant:
                value, new_quote_types = self._str_literal_helper(
                    value, quote_types=quote_types, escape_special_whitespace=True)
                if set(new_quote_types).isdisjoint(quote_types):
                    fallback_to_repr = True
                    break
                quote_types = new_quote_types
            elif '\n' in value:
                quote_types = [q for q in quote_types if q in _MULTI_QUOTES]
                assert quote_types
            new_fstring_parts.append(value)
        if fallback_to_repr:
            quote_types = ["'''"]
            new_fstring_parts.clear()
            for value, is_constant in fstring_parts:
                if is_constant:
                    value = repr('"' + value)
                    expected_prefix = '\'"'
                    assert value.startswith(expected_prefix), repr(value)
                    value = value[len(expected_prefix):-1]
                new_fstring_parts.append(value)
        value = ''.join(new_fstring_parts)
        quote_type = quote_types[0]
        self.write(f'{quote_type}{value}{quote_type}')

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, tuple):
            with self.delimit('(', ')'):
                self.items_view(self._write_constant, value)
        elif value is ...:
            self.write('...')
        else:
            if node.kind == 'u':
                self.write('u')
            self._write_constant(node.value)

    def visit_SetComp(self, node):
        with self.delimit('{', '}'):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_Set(self, node):
        if node.elts:
            with self.delimit('{', '}'):
                self.interleave(lambda: self.write(', '),
                                self.traverse, node.elts)
        else:
            self.write('{*()}')

    def visit_BinOp(self, node):
        operator = self.binop[node.op.__class__.__name__]
        operator_precedence = self.binop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            if operator in self.binop_rassoc:
                left_precedence = operator_precedence.next()
                right_precedence = operator_precedence
            else:
                left_precedence = operator_precedence
                right_precedence = operator_precedence.next()
            self.set_precedence(left_precedence, node.left)
            self.traverse(node.left)
            self.write(f' {operator} ')
            self.set_precedence(right_precedence, node.right)
            self.traverse(node.right)

    def visit_Call(self, node):
        self.set_precedence(Precedence.ATOM, node.func)
        self.traverse(node.func)
        with self.delimit('(', ')'):
            comma = False
            for e in node.args:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.traverse(e)

    def visit_Slice(self, node):
        if node.lower:
            self.traverse(node.lower)
        self.write(':')
        if node.upper:
            self.traverse(node.upper)
        if node.step:
            self.write(':')
            self.traverse(node.step)

    def visit_keyword(self, node):
        if node.arg is None:
            self.write('**')
        else:
            self.write(node.arg)
            self.write('=')
        self.traverse(node.value)

    def visit_match_case(self, node):
        self.fill('case ')
        self.traverse(node.pattern)
        if node.guard:
            self.write(' if ')
            self.traverse(node.guard)
        with self.block():
            self.traverse(node.body)

    def visit_MatchStar(self, node):
        name = node.name
        if name is None:
            name = '_'
        self.write(f'*{name}')

    def interleave(self, inter, f, seq):
        """Call f on each item in seq, calling inter() in between."""
        seq = iter(seq)
        try:
            f(next(seq))
        except StopIteration:
            pass
        else:
            for x in seq:
                inter()
                f(x)

    def delimit_if(self, start, end, condition):
        if condition:
            return self.delimit(start, end)
        else:
            return nullcontext()

    def visit_Module(self, node):
        self._type_ignores = {
            ignore.lineno: f'ignore{ignore.tag}' for ignore in node.type_ignores}
        self._write_docstring_and_traverse_body(node)
        self._type_ignores.clear()

    def visit_AnnAssign(self, node):
        self.fill()
        with self.delimit_if('(', ')', not node.simple and isinstance(node.target, Name)):
            self.traverse(node.target)
        self.write(': ')
        self.traverse(node.annotation)
        if node.value:
            self.write(' = ')
            self.traverse(node.value)

    def visit_Nonlocal(self, node):
        self.fill('nonlocal ')
        self.interleave(lambda: self.write(', '), self.write, node.names)

    def visit_ExceptHandler(self, node):
        self.fill('except*' if self._in_try_star else 'except')
        if node.type:
            self.write(' ')
            self.traverse(node.type)
        if node.name:
            self.write(' as ')
            self.write(node.name)
        with self.block():
            self.traverse(node.body)

    def visit_For(self, node):
        self._for_helper('for ', node)

    def visit_FormattedValue(self, node):

        def unparse_inner(inner):
            unparser = type(self)()
            unparser.set_precedence(Precedence.TEST.next(), inner)
            return unparser.visit(inner)
        with self.delimit('{', '}'):
            expr = unparse_inner(node.value)
            if expr.startswith('{'):
                self.write(' ')
            self.write(expr)
            if node.conversion != -1:
                self.write(f'!{chr(node.conversion)}')
            if node.format_spec:
                self.write(':')
                self._write_fstring_inner(
                    node.format_spec, is_format_spec=True)

    def visit_comprehension(self, node):
        if node.is_async:
            self.write(' async for ')
        else:
            self.write(' for ')
        self.set_precedence(Precedence.TUPLE, node.target)
        self.traverse(node.target)
        self.write(' in ')
        self.set_precedence(Precedence.TEST.next(), node.iter, *node.ifs)
        self.traverse(node.iter)
        for if_clause in node.ifs:
            self.write(' if ')
            self.traverse(if_clause)

    def visit_BoolOp(self, node):
        operator = self.boolops[node.op.__class__.__name__]
        operator_precedence = self.boolop_precedence[operator]

        def increasing_level_traverse(node):
            nonlocal operator_precedence
            operator_precedence = operator_precedence.next()
            self.set_precedence(operator_precedence, node)
            self.traverse(node)
        with self.require_parens(operator_precedence, node):
            s = f' {operator} '
            self.interleave(lambda: self.write(
                s), increasing_level_traverse, node.values)

    def visit_arg(self, node):
        self.write(node.arg)
        if node.annotation:
            self.write(': ')
            self.traverse(node.annotation)

    def visit_MatchSingleton(self, node):
        self._write_constant(node.value)

    def write(self, *text):
        """Add new source parts"""
        self._source.extend(text)

    def visit_Import(self, node):
        self.fill('import ')
        self.interleave(lambda: self.write(', '), self.traverse, node.names)

    def visit_Raise(self, node):
        self.fill('raise')
        if not node.exc:
            if node.cause:
                raise ValueError(f"Node can't use cause without an exception.")
            return
        self.write(' ')
        self.traverse(node.exc)
        if node.cause:
            self.write(' from ')
            self.traverse(node.cause)

    def visit_With(self, node):
        self.fill('with ')
        self.interleave(lambda: self.write(', '), self.traverse, node.items)
        with self.block(extra=self.get_type_comment(node)):
            self.traverse(node.body)

    def visit_Tuple(self, node):
        with self.delimit_if('(', ')', len(node.elts) == 0 or self.get_precedence(node) > Precedence.TUPLE):
            self.items_view(self.traverse, node.elts)

    def visit_alias(self, node):
        self.write(node.name)
        if node.asname:
            self.write(' as ' + node.asname)

    def get_raw_docstring(self, node):
        """If a docstring node is found in the body of the *node* parameter,
        return that docstring node, None otherwise.

        Logic mirrored from ``_PyAST_GetDocString``."""
        if not isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)) or len(node.body) < 1:
            return None
        node = node.body[0]
        if not isinstance(node, Expr):
            return None
        node = node.value
        if isinstance(node, Constant) and isinstance(node.value, str):
            return node

    def visit_TypeVar(self, node):
        self.write(node.name)
        if node.bound:
            self.write(': ')
            self.traverse(node.bound)
        if node.default_value:
            self.write(' = ')
            self.traverse(node.default_value)

    def visit_Starred(self, node):
        self.write('*')
        self.set_precedence(Precedence.EXPR, node.value)
        self.traverse(node.value)

    def visit_Continue(self, node):
        self.fill('continue')

    def visit_MatchClass(self, node):
        self.set_precedence(Precedence.ATOM, node.cls)
        self.traverse(node.cls)
        with self.delimit('(', ')'):
            patterns = node.patterns
            self.interleave(lambda: self.write(', '), self.traverse, patterns)
            attrs = node.kwd_attrs
            if attrs:

                def write_attr_pattern(pair):
                    attr, pattern = pair
                    self.write(f'{attr}=')
                    self.traverse(pattern)
                if patterns:
                    self.write(', ')
                self.interleave(lambda: self.write(', '), write_attr_pattern, zip(
                    attrs, node.kwd_patterns, strict=True))

    def visit_MatchOr(self, node):
        with self.require_parens(Precedence.BOR, node):
            self.set_precedence(Precedence.BOR.next(), *node.patterns)
            self.interleave(lambda: self.write(' | '),
                            self.traverse, node.patterns)

    def visit_ListComp(self, node):
        with self.delimit('[', ']'):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)


def nstr(node: ast.AST) -> str:
    return ast.unparse(node)


class CombinerVisitor(ast.NodeVisitor):
    """
    Visits modules and combines them into a single module

    Unique Traits:
        - all imports are turned into singletons 
        - finds all the constant assignments 
        - finds all the functions
        - finds all the classes
        - finds all the enums
        - determines if the import is from a local file and processes it
        - recursively determines imports and processes them
        - determines if the import is from a local directory and processes it
        - recursively determines from imports and processes them
        - finds all the other code
        - skips all if __name__ == __main__ statements 



    """

    def __init__(self):
        self.imports = []
        self.enums = []
        self.classes = []
        self.functions = []
        self.constants = []
        self.other_code = []
        self.local_files = [os.path.splitext(file)[0] for file in os.listdir(
            os.getcwd()) if file.endswith('.py') and os.path.isfile(file)]
        self.local_dirs = [d for d in os.listdir(
            os.getcwd()) if os.path.isdir(d)]
        for d in self.local_dirs:
            if d.startswith('.'):
                continue
            this_d = d.replace('.', os.sep)
            for sub in os.listdir(this_d):
                path = os.path.join(this_d, sub)
                if os.path.isdir(path):
                    self.local_dirs.append(path.replace(os.sep, '.'))

    def generic_visit(self, node: ast.AST):
        self.other_code.append(node)

    def _name_is_constant(self, node: ast.Name):
        return node.id.isupper()

    def _singleton_Import_is_local_file(self, node: ast.Import):
        return node.names[0].name in self.local_files

    def _reduce_Import(self, node: ast.Import) -> list[ast.Import]:
        return [ast.Import([imp]) for imp in node.names]

    def _visit_local_import(self, imp: ast.Import):
        """
        Visits a local import and processes its contents.

        This function is called when an import is detected that is a local file.
        It opens the file, parses its contents with the AST, and then visits the
        parsed contents to process them.  After visiting the contents, it removes
        the local file from the list of local files so that it isn't processed
        again.

        The filename is determined by taking the name of the import and adding
        .py to the end.  The file is assumed to be in the same directory as this
        file.
        """
        with open(os.path.join(os.path.dirname(__file__), f'{imp.names[0].name}.py'), 'r', encoding='utf-8') as f:
            structure = ast.parse(f.read(), filename=os.path.join(
                os.path.dirname(__file__), f'{imp.names[0].name}.py'))
        substack = CombinerVisitor()
        substack.visit(structure)
        self.merge(substack)
        self.local_files.remove(imp.names[0].name)

    def _visit_local_FromImport(self, imp: ast.ImportFrom):
        string = self._ImportFrom_to_string(imp)
        path = string.replace('.', os.sep) + '.py'
        with open(path, 'r', encoding='utf-8') as f:
            structure = ast.parse(f.read())
        substack = CombinerVisitor()
        substack.visit(structure)
        self.merge(substack)
        self.local_files.remove(string)

    def _ImportFrom_to_string(self, node: ast.ImportFrom) -> str:
        return f'{node.module}.{node.names[0].name}'

    def visit_FunctionDef(self, node):
        self.other_code.append(node)

    def _class_base_is_enum(self, node):
        if isinstance(node, ast.Name):
            return node.id in ['Enum', 'EnumMeta', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag']
        elif isinstance(node, ast.Attribute):
            return node.attr in ['Enum', 'EnumMeta', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag']
        elif isinstance(node, ast.Subscript):
            return node.value.id in ['Enum', 'EnumMeta', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag']
        elif isinstance(node, str):
            return node in ['Enum', 'EnumMeta', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag']
        return False

    def visit(self, node: ast.AST):
        method_name = 'visit_' + node.__class__.__name__
        if hasattr(self, method_name):
            getattr(self, method_name)(node)
        else:
            self.generic_visit(node)

    def visit_Module(self, node):
        for child in node.body:
            if isinstance(child, ast.If) and isinstance(child.test, ast.Compare) and isinstance(child.test.left, ast.Name) and (child.test.left.id == '__name__') and isinstance(child.test.comparators[0], ast.Name) and (child.test.comparators[0].id == '__main__'):
                continue
            else:
                self.visit(child)

    def visit_ImportFrom(self, node):
        if self._ImportFrom_to_string(node) in self.local_dirs:
            self._visit_local_FromImport(node)
        else:
            self.imports.append(node)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
            self.constants.append(node)
            return
        elif len(node.targets) > 1:
            index = []
            for idx, target in enumerate(node.targets):
                if isinstance(target, ast.Name) and self._name_is_constant(target):
                    index.append(idx)
            if index:
                self.constants.append(node)
                return
        self.other_code.append(node)

    def merge(self, visitor: 'CombinerVisitor'):
        self.imports.extend(visitor.imports)
        self.enums.extend(visitor.enums)
        self.classes.extend(visitor.classes)
        self.functions.extend(visitor.functions)
        self.constants.extend(visitor.constants)
        self.other_code.extend(visitor.other_code)

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.bases:
            for basenode in node.bases:
                if self._class_base_is_enum(basenode):
                    self.enums.append(node)
                    return
        self.other_code.append(node)

    def visit_Import(self, node):
        imps = self._reduce_Import(node)
        for imp in imps:
            if self._singleton_Import_is_local_file(imp):
                self._visit_local_import(imp)
            else:
                self.imports.append(imp)


class CombineWriter:

    def string(self):
        retv = ''
        if len(self.visitor.imports) != 0:
            for node in self.visitor.imports:
                retv += nstr(node) + '\n'
            retv += '\n\n\n'
        if len(self.visitor.constants) != 0:
            for node in self.visitor.constants:
                retv += ast.unparse(node) + '\n'
            retv += '\n\n\n'
        if len(self.visitor.enums) != 0:
            for node in self.visitor.enums:
                retv += ast.unparse(node) + '\n\n\n\n'
            retv += '\n\n\n'
        if len(self.visitor.classes) == 0:
            for node in self.visitor.classes:
                retv += ast.unparse(node) + '\n\n\n\n'
        if len(self.visitor.functions) == 0:
            for node in self.visitor.functions:
                retv += ast.unparse(node) + '\n\n\n'
            retv += '\n'
        if len(self.visitor.other_code) != 0:
            for node in self.visitor.other_code:
                retv += ast.unparse(node) + '\n'
            retv += '\n'
        retv += '\n\n\n'
        return retv

    def __init__(self, visitor: CombinerVisitor):
        self.visitor = visitor
        self.visitor.imports = list(
            {ast.dump(imp): imp for imp in visitor.imports}.values())


__all__ = ['combine_files_with_ast', 'run']
if __name__ == '__main__':
    source_code = '\nimport os\nimport sys\n\ndef example_function():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    example_function()\n'
    with open(__file__, 'r', encoding='utf-8', errors='ignore') as f:
        source_code = f.read()
    linter = SourceLinter(source_code)
    problems = linter.get_problems()
    SourceReader(source_code).react(problems)

