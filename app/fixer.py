import subprocess
from dataclasses import dataclass
import tokenize
from io import StringIO
from typing import Optional


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
        return -1  # No need to split if the line is already within the limit

    tokens = list(tokenize.generate_tokens(StringIO(line).readline))
    split_candidates = []

    for i, token in enumerate(tokens):
        tok_type, tok_string, start, end, _ = token

        # Only consider these token types for splitting
        if tok_type == tokenize.OP and tok_string in {',', '+', '-', '*', '/', '%', '(', '[', '{'}:
            split_candidates.append(start[1])

    # Remove candidates that occur within strings or comments
    for token in tokens:
        tok_type, _, start, _, _ = token
        if tok_type in {tokenize.STRING, tokenize.COMMENT}:
            col = start[1]
            split_candidates = [pos for pos in split_candidates if pos < col]

    # Filter candidates to ensure both split parts fit within max_length
    split_candidates = [pos for pos in split_candidates if pos <= max_length and len(line) - pos <= max_length]

    # Return the rightmost valid split position, or -1 if none
    return max(split_candidates) if split_candidates else -1



@dataclass
class Problem:
    file: str
    line: int
    char: int
    code: str
    message: str
    error: str

    def fromLine(line: str) -> Optional["Problem"]:
        data   = line.split(":")
        if len(data) > 3:
            return Problem(data[0].strip(), 
                           int(data[1].strip()), 
                           int(data[2].strip()), 
                           data[3].strip(), 
                           data[4].strip()[0:data[4].index("(")-1], 
                           data[4].strip().split("(")[1].split(")")[0].strip())
        else:
            return None
                        

class ProblemFixer:
    def __init__(self, source:str) -> None:
        self.source: str = source
        self.problems: list[Problem] = []
    
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

        tempfile = "temp_file.py"
        with open(tempfile, "w", encoding="utf-8") as f:
            f.write(self.source)

        try:
            result = subprocess.run(["pylint", tempfile], capture_output=True, text=True)
            output = result.stdout  # Return the linter's output
            problems = []
            lines = output.split("\n")
            for line in lines:
                prob = Problem.fromLine(line)
                if prob is not None:
                    problems.append(prob)
        except Exception as e:
            return 
        finally:
            self.problems = problems

    def setsrc(self, source:str):
        self.source = source
    
    def FIX_C0304(self, problem:Problem):
        if not self.source.endswith("\n"):
            self.source += "\n"
            print("Fixed missing newline at end of file")
        else:
            print("No missing newline at end of file")

    def FIX_C0303(self, problem:Problem):
        line_index = problem.line
        lines = self.source.split("\n")
        
        if line_index < 0 or line_index >= len(lines):
            print("Line index out of range")
            return
        
        lines[line_index] = lines[line_index].rstrip()
        self.source = "\n".join(lines)
        print("Fixed trailing whitespace on line ", line_index)
        lines[line_index] = lines[line_index].rstrip()
        self.source = "\n".join(lines)
        print("Fixed trailing whitespace on line ", line_index)

    def _indentation(self, line:str):
        return len(line) - len(line.lstrip())
    
    def FIX_C0103(self, problem:Problem) -> None:
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
                print("Problem does not contain a line number")
                return

            lines = self.source.split("\n")

            if line_index >= len(lines) or line_index < 0:
                print("Line index out of range")
                return

            line = lines[line_index]

            if '=' not in line:
                print("No assignment found on line ", line_index)
                return

            left, right = line.split('=', 1)
            left = left.strip()
            left = left.upper()

            lines[line_index] = f"{left}={right}"
            self.source = "\n".join(lines)
            print("Fixed constant name on line ", line_index)

        except Exception as e:
            print("Error fixing constant name: ", e)
        

    def FIX_C0301(self, problem:Problem):
        """
        Fixes a line that is too long.
        """
        maxlinelength = 78
        line_index = problem.line
        lines = self.source.split("\n")
        lines[line_index] = lines[line_index].rstrip()
        indent = ' ' * self._indentation(lines[line_index])
        nextline = indent + lines[line_index][maxlinelength:]
        thisline = lines[line_index][:maxlinelength] + "\\"
        
        lines[line_index] = thisline
        lines.insert(line_index + 1, nextline)
        
        self.source = "\n".join(lines)
        print("Fixed line length on line ", line_index)
        nextline = indent + lines[line_index][maxlinelength:]
        thisline = lines[line_index][:maxlinelength] + "\\"
        
        self.source = "\n".join(lines)
        print("Fixed line length on line ", line_index)
    
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
            lines = self.source.split("\n")

            if line_index >= len(lines) or line_index < 0:
                print("Line index out of range")
                return

            line = lines[line_index]

            if not line.startswith("    ") and not line.startswith("\t"):
                print("No redefinition found on line ", line_index)
                return

            if '"""' in line or "'''" in line:
                print("No redefinition found on line ", line_index)
                return

            lines[line_index] = ""
            self.source = "\n".join(lines)
            print("Fixed redefining name on line ", line_index)

        except Exception as e:
            print(f"An error occurred while fixing redefining name: {e}")
    
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
            print(f"An error occurred while fixing unnecessary elif: {e}")
    
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
            print(f"An error occurred while fixing unnecessary elif: {e}")
    
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
                # If the line number is None, return and do nothing
                print("Line number is None")
                return

            lines = self.source.splitlines()

            if line_number >= len(lines) or line_number < 0:
                # If the line number is out of range, return and do nothing
                print("Line index out of range")
                return

            line = lines[line_number]
            if 'def' not in line:
                # If the line doesn't contain the word "def", return and do nothing
                print("No defining found on line ", line_number)
                return

            # Replace the line with the new line
            lines[line_number] = line.replace('def', 'async def')
            # Join the lines back into the source code
            self.source = '\n'.join(lines)

        except IndexError:
            # Print an error message if the line number is out of range
            print(f"Line index out of range: {line_number}")
        except Exception as e:
            # Print an error message if an exception occurs
            print(f"An error occurred while fixing redefining name: {e}")

    def FIX_C0115(self, problem: Problem) -> None:
        """
        Fixes a missing function or method docstring.
        """
        try:
            line_number = problem.line
            if line_number is None:
                print("Line number is None")
                return

            lines = self.source.splitlines()

            if line_number >= len(lines) or line_number < 0:
                print("Line index out of range")
                return

            line = lines[line_number]
            if '"""' in line:
                print("Docstring already present on line", line_number)
                return

            docstring = '"""Add docstring here"""'
            lines.insert(line_number, docstring)
            self.source = '\n'.join(lines)
            print("Added docstring on line", line_number)

        except Exception as e:
            print(f"An error occurred while fixing missing docstring: {e}")

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

        # Get the list of problems from the linter
        self.lint()

        # If there are no problems, print a message and return
        if len(self.problems) == 0:
            print("No problems to fix")
            return

        # Iterate over each problem in the list
        for problem in self.problems:
            # Get the method name for the FIX_<code> method
            method = "FIX_" + problem.code.upper()

            # Check if the method exists
            if not hasattr(self, method):
                # If it doesn't, skip this problem
                continue
            else:
                # If it does, get the method from the class
                method = getattr(self, method)
                # Call the method with the problem as an argument
                method(problem)
    
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

        # Get the list of problems from the linter
        self.lint()

        # If there are no problems, print a message and return
        if len(self.problems) == 0:
            print("No problems to fix")
            return

        # Iterate over each problem in the list
        for problem in self.problems:
            # Get the method name for the FIX_<code> method
            method = "FIX_" + problem.code.upper()

            # Check if the method exists
            if not hasattr(self, method):
                # If it doesn't, skip this problem
                continue
            else:
                # If it does, get the method from the class
                method = getattr(self, method)
                # Call the method with the problem as an argument
                method(problem)

    #def recreate_problems(self, source: str) -> list:
    #    """
    #    Recreates the list of problems from the source code.
    #
    #    Args:
    #        source (str): The source code to lint.
    #
    #    Returns:
    #        list: A list of Problem objects.
    #    """
    #    self.setsrc(source)
    #    self.lint()
    #    return self.problems
