import ast
import os


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
        self.imports    = []
        self.enums      = []
        self.classes    = []
        self.functions  = []
        self.constants  = []

        self.other_code = []

        self.local_files = [
            os.path.splitext(file)[0]
            for file in os.listdir(os.getcwd()) 
            if file.endswith(".py") and os.path.isfile(file)
        ]

        self.local_dirs = [
            d for d in os.listdir(os.getcwd())
            if os.path.isdir(d)
        ]
        # Recursively add subdirectories
        for d in self.local_dirs:
            if d.startswith("."): 
                continue
            
            this_d = d.replace(".", os.sep)
            for sub in os.listdir(this_d):
                path = os.path.join(this_d, sub)
                if os.path.isdir(path):
                    self.local_dirs.append(path.replace(os.sep, "."))

    def merge(self, visitor: 'CombinerVisitor'):
        self.imports.extend(visitor.imports)
        self.enums.extend(visitor.enums)
        self.classes.extend(visitor.classes)
        self.functions.extend(visitor.functions)
        self.constants.extend(visitor.constants)
        self.other_code.extend(visitor.other_code)

    def generic_visit(self, node: ast.AST):
        self.other_code.append(node)

    def visit(self, node: ast.AST):
        # Check if a specific visit method exists for the node's class type
        method_name = "visit_" + node.__class__.__name__
        
        if hasattr(self, method_name):
            # Call the specific visit method for the node's class type
            getattr(self, method_name)(node)
        else:
            # Fallback to the generic visit method if no specific method exists
            self.generic_visit(node)
    

    def _name_is_constant(self, node: ast.Name):
        return node.id.isupper()

    def visit_Assign(self, node: ast.Assign):

        # Check if the left side is a constant
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

    def _singleton_Import_is_local_file(self, node: ast.Import):
        return node.names[0].name in self.local_files
        
    def _reduce_Import(self, node: ast.Import) -> list[ast.Import]:
        return [ast.Import([imp]) for imp in node.names]
    
    def visit_Module(self, node):
        for child in node.body:
            # Ignore if it's an '__name__' == __main__:
            if isinstance(child, ast.If)  and isinstance(child.test, ast.Compare) and isinstance(child.test.left, ast.Name) \
                and child.test.left.id == "__name__"  and isinstance(child.test.comparators[0], ast.Name) \
                and child.test.comparators[0].id == "__main__":
                    continue
            else:
                self.visit(child)

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
        with open(os.path.join(os.path.dirname(__file__), f"{imp.names[0].name}.py"), "r", encoding="utf-8") as f:
            # Open the file and read it into a string.
            structure = ast.parse(f.read(), filename=os.path.join(os.path.dirname(__file__), f"{imp.names[0].name}.py"))
        #Visit the parsed contents.
        substack = CombinerVisitor()
        substack.visit(structure)
        self.merge(substack)
        self.local_files.remove(imp.names[0].name)

    def _visit_local_FromImport(self, imp: ast.ImportFrom):
        string = self._ImportFrom_to_string(imp)
        path = string.replace(".", os.sep) + ".py"
        with open(path, "r", encoding="utf-8") as f:
            structure = ast.parse(f.read())
        substack = CombinerVisitor()
        substack.visit(structure)
        self.merge(substack)
        self.local_files.remove(string)


    def visit_Import(self, node):
        # Reduce the import node to a list of individual import statements
        imps = self._reduce_Import(node)

        # Iterate over each import statement
        for imp in imps:
            # Check if the import is a local file
            if self._singleton_Import_is_local_file(imp):
                # If it is a local file, visit the file to process its contents
                self._visit_local_import(imp)
            else:
                # If it is not a local file, add the import to the list of imports
                self.imports.append(imp)

    def _ImportFrom_to_string(self, node: ast.ImportFrom) -> str:
        return f"{node.module}.{node.names[0].name}"

    def visit_ImportFrom(self, node):
        if self._ImportFrom_to_string(node) in self.local_dirs:
            self._visit_local_FromImport(node)
        else:
            self.imports.append(node)

    def visit_FunctionDef(self, node):
        self.other_code.append(node)

    def _class_base_is_enum(self, node):
        if isinstance(node, ast.Name):
            return node.id in ["Enum", "EnumMeta", "IntEnum", "StrEnum", "Flag", "IntFlag"]
        elif isinstance(node, ast.Attribute):
            return node.attr in ["Enum", "EnumMeta", "IntEnum", "StrEnum", "Flag", "IntFlag"]
        elif isinstance(node, ast.Subscript):
            return node.value.id in ["Enum", "EnumMeta", "IntEnum", "StrEnum", "Flag", "IntFlag"]
        elif isinstance(node, str):
            return node in ["Enum", "EnumMeta", "IntEnum", "StrEnum", "Flag", "IntFlag"]
        
        return False
    
    def visit_ClassDef(self, node: ast.ClassDef):
        
        if node.bases:
            for basenode in node.bases:
                if self._class_base_is_enum(basenode):
                    self.enums.append(node)
                    return 
        
        self.other_code.append(node)

class CombineWriter:
    def __init__(self, visitor: CombinerVisitor):
        self.visitor = visitor
        self.visitor.imports = list({ast.dump(imp): imp for imp in visitor.imports}.values())


    def string(self):
        retv = ""

        if len(self.visitor.imports) != 0:

            for node in self.visitor.imports:
                retv += nstr(node) + "\n"
        
            retv += "\n\n\n"

        if len(self.visitor.constants) != 0:

            for node in self.visitor.constants:
                retv += ast.unparse(node) + "\n"
        
            retv += "\n\n\n"

        if len(self.visitor.enums) != 0:

            for node in self.visitor.enums:
                retv += ast.unparse(node) + "\n\n\n\n"
        
            retv += "\n\n\n"

        if len(self.visitor.classes) == 0:
            
            for node in self.visitor.classes:
                retv += ast.unparse(node) + "\n\n\n\n"
        
        if len(self.visitor.functions) == 0:
            
            for node in self.visitor.functions:
                retv += ast.unparse(node) + "\n\n\n"
        
            retv += "\n"

        if len(self.visitor.other_code) != 0:
            for node in self.visitor.other_code:
                retv += ast.unparse(node) + "\n"
        
            retv += "\n"

        retv += "\n\n\n"

        return retv

