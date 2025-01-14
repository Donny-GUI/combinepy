import ast
import autopep8


def is_if_name_is_main(stmt: ast.If):
    string = ast.unparse(stmt.test)
    items = ['__name__', '__main__', "=="]
    for item in items:
        if item not in string:
            return False

    return True




class FeatureApplier:
    def __init__(self, source:str):
        self.source = source
        self.tree = ast.parse(source)
        self._format = None
    
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
        
        pass 

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
                



