import subprocess
import shutil 
from dataclasses import dataclass


@dataclass
class Problem:
    file: str
    line: int
    char: int
    code: str
    message: str
    error: str

    def __str__(self):
        return f"{self.file}:{self.line}:{self.char}:{self.code}:{self.message}:{self.error}"
    
    def __repr__(self):
        return f"{self.file}:{self.line}:{self.char}:{self.code}:{self.message}:{self.error}"

    @classmethod
    def fromLine(cls, line:str):
        data = line.split(":")
        if len(data) > 3:
            d = {"file": data[0].strip(), 
                 "line": int(data[1].strip()), 
                 "char": int(data[2].strip()), 
                 "code": data[3].strip(),
                 "message": data[4].strip()[0:data[4].index("(")-1],
                 "error": data[4].strip().split("(")[1].split(")")[0].strip(),
                }
            return cls(**d)
        else:
            return None


class SourceLinter:
    def __init__(self, source: str):
        """
        Initialize the SourceLinter with the given source code.
        
        Args:
            source (str): The source code to be linted.
        """
        self.source = source
    
    def write(self):
        """
        Writes the source code to a temporary file.
        
        The temporary file is named "temp_file.py" and is placed in the current working directory.
        The file is written in UTF-8 encoding.
        """
        with open("temp_file.py", "w", encoding="utf-8") as f:
            f.write(self.source)

    def lint(self) -> str:
        """
        Runs pylint on the source code and returns the output.

        The source code is saved to a temporary file and then pylint is run as a subprocess.
        The output from pylint is captured and returned as a string.

        If an exception occurs while running pylint, the function returns a string containing the exception message.

        Returns:
            str: The output from pylint or an error message if an exception occurred.
        """

        temp_file = "temp_file.py"
        # Save the source code to a temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(self.source)

        # Run the linter as a subprocess
        try:
            result = subprocess.run(["pylint", temp_file], capture_output=True, text=True)
            return result.stdout  # Return the linter's output
        except Exception as e:
            return f"Linter failed: {e}"

    def parse_pylint_output(self, output: str) -> list[Problem]:
        """
        Parses the output from pylint and returns a string representation of the issues.

        Args:
            output (str): The output from pylint.

        Returns:
            str: A string representation of the issues.
        """
        problems = []
        lines = output.split("\n")
        for line in lines:
            data = line.split(":")
            if len(data) > 3:
                problems.append(Problem.fromLine(line))
                
        return problems

    def get_problems(self) -> list[Problem]:
        """
        Executes the linter on the source code and returns a list of linting issues.

        The source code is first saved to a temporary file, and the linter is executed
        as a subprocess. The output is parsed to extract the issues, which are returned
        as a list of Problem objects.

        Returns:
            list[Problem]: A list of Problem objects representing the linting issues.
        """

        temp_file = "temp_file.py"
        # Save the source code to a temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(self.source)

        # Run the linter as a subprocess
        try:
            result = subprocess.run(["pylint", temp_file], capture_output=True, text=True)
            output =result.stdout  # Return the linter's output
        except Exception as e:
            return f"Linter failed: {e}"

        problems = []
        lines = output.split("\n")
        for line in lines:
            data = line.split(":")
            if len(data) > 3:
                problems.append(Problem.fromLine(line))
                
        return problems


    def lint_summary(self, linter="pylint") -> str:
        """
        Provides a concise summary of linting results.

        Args:
            linter (str): The linter to use ('pylint' or 'flake8').

        Returns:
            str: A brief summary of linting results, including issue count.
        """
        output = self.lint(linter)
        if linter == "pylint":
            output = self.parse_pylint_output(output)
            for issue in output:
                print(issue)










def red(text):
    return f"\033[91m{text}\033[0m"
def background_red(text):
    return f"\033[41m{text}\033[0m"

def redline(text):
    print(f"\033[91m{text}\033[0m")

def green(text):
    return f"\033[92m{text}\033[0m"

def greenline(text):
    print(f"\033[92m{text}\033[0m")

def blue(text):
    return f"\033[94m{text}\033[0m"

def blueline(text):
    print(f"\033[94m{text}\033[0m")

def dim(text):
    return f"{DIMBLACK}{text}\033[0m"

FATAL   = "❌"
WARN    = "⛔"
GOOD    = "✅"
CONV    = "🆗"
RECM    = "❎"
WALL    = "|"
DIMBLACK = "\033[2;30m"

class SourceReader:

    def __init__(self, source: str):
        self.source = source
        self.problems: list[Problem] = []
    

    def get_icon(self, problem: Problem):
        if isinstance(problem, type(None)):
            return GOOD
            
        if problem.code.startswith("E"):
            return FATAL
        elif problem.code.startswith("W"):
            return WARN
        elif problem.code.startswith("C"):
            return CONV
        elif problem.code.startswith("R"):
            return RECM
        else:
            return GOOD

    def react(self, problems:list[Problem]):
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
        # Get the width of the terminal window
        
        # Get the line numbers of all the problems
        problem_lines = [problem.line for problem in problems]
        
        # Read the source code file line by line
        with open(problems[0].file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        # Find the maximum length of the line number and the maximum length of the line
        max_length = len(str(len(lines)))
        max_line_length = max([len(line) for line in lines]) + 5
        max_terminal_width = shutil.get_terminal_size().columns

        additional_text = ""
        
        # Iterate over each line in the source code
        for index, line in enumerate(lines):
            # Remove trailing whitespace from the line
            line = line.rstrip()
            
            # Increment the line number to be 1-based instead of 0-based
            index = index + 1
            
            # If this line has a problem, print it in red and add the error message
            if index in problem_lines:
                # Get the problem from the list of problems
                problem = [p for p in problems if p.line == index][0]
                icon = self.get_icon(problem)
                # If the problem is on the first character of the line, print the whole line in red
                if problem.char == 0:
                    line = f"{icon+WALL}"+ str(index).ljust(max_length) + WALL + red(line)
                    line = line.ljust(max_line_length) + WALL + blue(problem.code) + " " + problem.message 
                # Otherwise, print the line up to the problem in green, the problem in red, and the rest of the line in green
                else:
                    line = f"{icon+WALL}" +  str(index).ljust(max_length) + WALL + line[:problem.char] \
                        + background_red(line[problem.char:])
                    line = line.ljust(max_line_length) + WALL \
                        + blue(problem.code) + " " + problem.message
            # Otherwise, print the line in green and add a blank line
            else:
                icon = GOOD
                line = f"{icon+WALL}" + str(index).ljust(max_length) + WALL + dim(line)
                line = line.ljust(max_line_length) + WALL

            # Print the line
            if len(line) > max_terminal_width:
                # put the rest of the line on the end of the next line 
                print(line[:max_terminal_width])
                additional_text = "     " +line[max_terminal_width:]
                
            else:
                if additional_text != "":
                    line = line + additional_text
                    additional_text = ""

                print(line)


# Example Usage
if __name__ == "__main__":
    source_code = """
import os
import sys

def example_function():
    print("Hello, World!")

if __name__ == "__main__":
    example_function()
"""
    with open(__file__, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()
    
    linter = SourceLinter(source_code)
    problems = linter.get_problems()
    SourceReader(source_code).react(problems)
    