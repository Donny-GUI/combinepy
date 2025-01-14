import ast 
import autopep8


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
            # Identify relative imports (level > 0)
            if node.level > 0:
                # Calculate the base module path
                base_parts = base_module.split(".")
                absolute_path = base_parts[:-node.level]  # Strip `level` parts from the base module
                if node.module:
                    absolute_path.append(node.module)
                absolute_import = f"from {'.'.join(absolute_path)} import {', '.join(a.name for a in node.names)}"
                relative_imports.append(absolute_import)
            self.generic_visit(node)

    # Find all relative imports
    finder = RelativeImportFinder()
    finder.visit(ast_module)

    # Convert AST back to source code
    original_source = ast.unparse(ast_module)

    # Append the resolved relative imports to the source code
    resolved_imports = "\n".join(relative_imports)
    if resolved_imports:
        return f"{original_source}\n# Resolved Relative Imports:\n{resolved_imports}"
    else:
        return original_source


def is_if_name_is_main(node):
    string = ast.unparse(node.test)
    items = ['__name__', '__main__', "=="]
    for item in items:
        if item not in string:
            return False

    return True


class FeatureApplier:
    def __init__(self, source:str):
        self.source: str = source
        self.tree = ast.parse(source)
        self._format: str = None
    
    def _rewriteSource(self):
        """Updates the source code string from the current AST."""

        self.source = ast.unparse(self.tree)
        match self._format:
            case None:
                return 
            case "autopep8":
                self.format_with_autopep8()
            case "black":
                self.format_with_black()
    
    def _rewriteTree(self):
        """Parses the current source code into an abstract syntax tree (AST) and updates the tree attribute."""

        self.tree = ast.parse(self.source)
    
    def _body(self, body:list[ast.AST]):
        """
        Sets the body of the current AST to the given list of nodes. 
        This is a lower-level version of the other methods in this class, 
        which create nodes or modify the tree for you.
        """
        setattr(self.tree, "body", body)
    
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

            if (isinstance(node, ast.If) and is_if_name_is_main(node)):
                ifname_body.extend(node.body)
                has_main = True
            else:
                keeping.append(node)

        if has_main == False:
            return

        ifn = ast.If(
            test=ast.Compare(
                left=ast.Name(
                    id='__name__', 
                    ctx=ast.Load()
                    ), 
                ops=[ast.Eq()], 
                comparators=[ast.Constant('__main__')]
                ), 
            body=ifname_body
            )

        keeping.append(ifn)
        self._body(keeping)
        self._rewriteSource()
        

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

                module = f"{node.module}." if node.module else ""
                for alias in node.names:
                    full_name = f"{module}{alias.name or alias.asname}"
                    imports[full_name] = node.lineno
        
        used_names = set()

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            if isinstance(node, ast.ClassDef) and node.bases:
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        used_names.add(base.id)
        
        def name_contained_in_module(name:str, module:str) -> bool:
            return module.find(name) != -1


        unused = {name: lineno for name, lineno in imports.items() if name not in used_names}
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
    
    def format_with_autopep8(self):
        self.source = autopep8.fix_code(self.source)
        self.tree = ast.parse(self.source)
        self._format = 'autopep8'

    def big_lists_one_per_line(self):
        """
        takes lists that are long and puts each item on its own line
        """
        pass 

    def private_methods_first(self):
        """
        for classes, write private methods first before public methods
        """
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        if subnode.name.startswith("_") == False:
                            node.body.remove(subnode)
                            node.body.insert(-1, subnode)
        
        self._rewriteSource()

    def class_assigns_before_any_method(self):
        """
        Reorders class assignments to appear before any methods in the class body.

        This function ensures that all assignments within a class definition are 
        placed before any method definitions. This can help in maintaining a 
        consistent structure within the class, making it easier to understand the 
        class's attributes and their initial values before diving into the methods.
        """
        def rearrange_class_assignments(node: ast.ClassDef):
            b:list[ast.AST] = node.body 
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

    def alphabetize_imports(self):
        """
        alphabetize imports
        """
        
        # Initialize 'firstindex' to the length of the AST body.
        # This will hold the index of the first import statement found.
        
        firstindex = len(self.tree.body)

        # Loop through each node in the AST body to find the first import statement.
        for index, node in enumerate(self.tree.body):
            # Check if the current node is an 'Import' or 'ImportFrom' node.
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Store the index of the first import statement and break the loop.
                firstindex = index
                break

        # Initialize an empty list to hold the unparsed import statements.
        imps = []

        # Iterate over the nodes starting from the first import statement found.
        for index, node in enumerate(self.tree.body[firstindex:]):
            # Check if the current node is an 'Import' or 'ImportFrom' node.
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Unparse the AST node to a string representation and append it to 'imps'.
                imps.append(ast.unparse(node))
                # Remove the import node from the AST body.
                self.tree.body.remove(node)

        # Sort the list of import statements alphabetically.
        imps.sort()
        # Parse the sorted import strings back into AST nodes.
        imps = [ast.parse(imp).body[0] for imp in imps]

        # Prepend the sorted import nodes to the AST body.
        self.tree.body = imps + self.tree.body

        # Rewrite the source code from the modified AST.
        self._rewriteSource()

    def alphabetize_methods(self):

        pass 

    def main_function_is_last(self):

        mainfunc = None
        lastindex = 0
        c = 0
        
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                lastindex = c
                if node.name == "main":
                    mainfunc = node
            c += 1

        if mainfunc != None:
            self.tree.body.remove(mainfunc)
            self.tree.body.insert(lastindex, mainfunc)
            self._rewriteSource()
    
    def _name_is_constant(self, name:ast.Name):
        return name.id.isupper()
    
    def _assign_is_constant(self, node:ast.Assign):
        return len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0])

    def group_constants_together(self):

        """
        Groups all constant assignments (assignments to uppercase names) together

        This function will go through the Abstract Syntax Tree (AST) and find the first constant assignment. 
        It will then group all following constant assignments together, and move them all to the front of the file.
        """
        
        # Find the index of the first constant assignment in the AST
        first_constant = len(self.tree.body)
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                # Check if the assignment is a constant assignment
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
                    first_constant = self.tree.body.index(node)
                    break
        # Create a list of all the constant assignments
        cons = []
        for node in self.tree.body[first_constant:]:
            if isinstance(node, ast.Assign):
                # Check if the assignment is a constant assignment
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self._name_is_constant(node.targets[0]):
                    cons.append(node)
                else:
                    break
            else:
                break
        # Rearrange the AST so that all constant assignments are at the front of the file
        self.tree.body = self.tree.body[:first_constant] + cons + self.tree.body[first_constant:]
        # Rewrite the source code based on the new AST
        self._rewriteSource()
    
    def remove_duplicate_functions(self):

        functions = []

        for node in self.tree.body:
            
            if isinstance(node, ast.FunctionDef):

                string = ast.unparse(node)

                if string in functions:
                    self.tree.body.remove(node)
                
                functions.append(string)

        self._rewriteSource()

    def remove_duplicate_classes(self):

        classes = []

        for node in self.tree.body:
            
            if isinstance(node, ast.ClassDef):
                if node in classes:
                    self.tree.body.remove(node)
                
                classes.append(node)

        self._rewriteSource()
    

    def remove_duplicate_constants(self):
        """
        Removes any duplicate constant assignments from the source code.

        This method goes through the Abstract Syntax Tree (AST) and finds all constant assignments.
        It then checks if any of the constant assignments are duplicates.  If a duplicate is found,
        it is removed from the AST.  Finally, the source code string is updated from the modified AST.
        """

        # Initialize an empty list to hold the unique constant assignments.
        constants = []
        body = self.tree.body.copy()

        # Iterate over each node in the AST body.
        for node in self.tree.body:
            
            # Check if the current node is a constant assignment.
            if isinstance(node, ast.Assign) and self._assign_is_constant(node) == True:

                # Convert the constant assignment node to a string representation.
                string = ast.unparse(node)

                # Check if the string representation is already in the list of constants.
                if string in constants:
                    # If it is, remove the node from the AST body.
                    body.remove(node)
                
                # Add the string representation to the list of constants.
                constants.append(string)
                
        self._body(body)
        # Rewrite the source code string based on the modified AST.
        self._rewriteSource()
    
    def remove_relative_imports(self):
        # Iterate over all nodes in the body of the AST
        for node in self.tree.body:
            # Check if the node is an import statement using 'from ... import ...'

            if isinstance(node, ast.ImportFrom):
                # Check if the import is relative (level is not zero)
                if node.level != 0:
                    # Remove the relative import from the AST
                    self.tree.body.remove(node)

        # Rewrite the source code string from the modified AST
        self._rewriteSource()
    
    
    


                



