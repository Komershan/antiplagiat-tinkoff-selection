from collections import defaultdict
import ast
from primitives import *


class Parser:

    def get_max_nesting(self, current_node: ast.AST):
        result = 0

        for node in ast.iter_child_nodes(current_node):
            result = max(result, self.get_max_nesting(node))

        if isinstance(current_node, (ast.For, ast.While)):
            result += 1

        return result

    def get_lex_count(self, current_node: ast.AST, count_lex: tuple[ast.AST] = ()):
        result = {}

        for node in ast.iter_child_nodes(current_node):
            for lexem, count in self.get_lex_count(node, count_lex).items():
                if not (lexem in result):
                    result.update({lexem: count})
                else:
                    result[lexem] += count

        key = f"{type(current_node)}"

        if isinstance(current_node, count_lex):
            if not (key in result):
                result.update({key: 1})
            else:
                result[key] += 1

        return result

    @staticmethod
    def get_includes(self, parsed: ast.AST):
        result = set()
        for node in ast.iter_child_nodes(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if not hasattr(alias, 'name'):
                        continue
                    for elem in alias.name.split('.'):
                        result.add(elem)
                if isinstance(node, ast.ImportFrom):
                    if not hasattr(node, 'module'):
                        continue
                    for elem in node.module.split('.'):
                        result.add(elem)
        return result

    def go_around_variables(self, current_node: ast.AST, change=False):
        result = set()

        for node in ast.iter_child_nodes(current_node):
            if not change:
                result |= self.go_around_variables(node, change)
            else:
                node = self.go_around_variables(node, change)

        if isinstance(current_node, ast.Assign):
            if isinstance(current_node.targets[0], ast.Tuple):
                for index in range(len(current_node.targets[0].elts)):
                    if not hasattr(current_node.targets[0].elts[index], 'id'):
                        continue
                    if not change:
                        result.add(current_node.targets[0].elts[index].id)
                    else:
                        current_node.targets[0].elts[index].id = "VAR"
            else:
                for index in range(len(current_node.targets)):
                    if not hasattr(current_node.targets[index], 'id'):
                        continue
                    if not change:
                        result.add(current_node.targets[index].id)
                    else:
                        current_node.targets[index].id = "VAR"

        if not change:
            return result
        else:
            return current_node

    def go_around_classes(self, current_node: ast.AST, change=False):

        result = set()

        for node in ast.iter_child_nodes(current_node):
            if not change:
                result |= self.go_around_classes(node, change)
            else:
                node = self.go_around_classes(node, change)

        if isinstance(current_node, ast.ClassDef):
            if not change:
                result.add(current_node.name)
            else:
                current_node.name = "CLASS"

        if not change:
            return result
        else:
            return current_node

    def go_around_functions(self, current_node: ast.AST, change=False):

        result = set()

        for node in ast.iter_child_nodes(current_node):
            if not change:
                result |= self.go_around_functions(node, change)
            else:
                node = self.go_around_functions(node, change)

        if isinstance(current_node, ast.FunctionDef):
            if not change:
                result.add(current_node.name)
            else:
                current_node.name = "FUNC"

        if not change:
            return result
        else:
            return current_node

    def go_around_all_names(self, current_node: ast.AST, change=False):

        result = set()

        for node in ast.iter_child_nodes(current_node):
            if not (change):
                result |= self.go_around_all_names(node, change)
            else:
                node = self.go_around_all_names(node, change)

        if hasattr(current_node, 'name'):
            if not (change):
                current_node.name = "ALL"
            else:
                result.add(current_node.name)

        if not change:
            return result
        else:
            return current_node

    def levenshtein_distance(self, first: str, second: str):
        distance = MyMatrix.scalars(len(first) + 1, len(second) + 1, 0)

        for i in range(0, len(first) + 1):
            for j in range(0, len(second) + 1):
                if i == 0:
                    distance[(i, j)] = j
                elif j == 0:
                    distance[(i, j)] = i
                else:
                    distance[(i, j)] = min(distance[(i - 1, j)] + 1, distance[(i, j - 1)] + 1,
                                           distance[(i - 1, j - 1)] + int(first[i - 1] == second[j - 1]))

        return distance[(len(first), len(second))]

    def calc_vocab_similarity(self, first_vocab: set[str], second_vocab: set[str]):
        max_similarity = {}

        for x in first_vocab:
            max_similarity.update({x: 0})

            for y in second_vocab:
                if not (y in max_similarity):
                    max_similarity.update({y: 0})

                max_similarity[x] = max(max_similarity[x], self.longest_common_subsequence(self, x, y))
                max_similarity[y] = max(max_similarity[y], self.longest_common_subsequence(self, x, y))

        return sum(max_similarity.values()) / max(len(max_similarity.items()), 1)

    @staticmethod
    def longest_common_subsequence(self, first: str, second: str):

        distance = MyMatrix.scalars(len(first) + 1, len(second) + 1, 0)

        for i in range(len(first) + 1):
            for j in range(len(second) + 1):
                if (i == 0 or j == 0):
                    distance[(i, j)] = 0
                else:
                    distance[(i, j)] = max(distance[(i - 1, j)], distance[(i, j - 1)])
                    if first[i - 1] == second[j - 1]:
                        distance[(i, j)] = max(distance[(i, j)], distance[(i - 1, j - 1)] + 1)

        return distance[(len(first), len(second))]

    def normalize_code(self, code: str):
        code_ast = ast.parse(code)
        code_ast = self.go_around_variables(code_ast, True)
        code_ast = self.go_around_functions(code_ast, True)
        code_ast = self.go_around_classes(code_ast, True)
        code_ast = self.go_around_all_names(code_ast, True)

        code = ast.unparse(code_ast)

        needs_to_replace = ['\n', ' ', '{', '}', ':', '(', ')', '"', "'"]

        for value in needs_to_replace:
            code = code.replace(value, '')

        return code

    def get_info(self, first_code: str, second_code: str):
        first_ast = ast.parse(first_code)
        second_ast = ast.parse(second_code)
        params = {}

        params.update(
            {'common_includes': len(self.get_includes(self, first_ast) & self.get_includes(self, second_ast))})

        first_variables = self.go_around_variables(first_ast)
        second_variables = self.go_around_variables(second_ast)
        params.update({'var_labels_similarity': self.calc_vocab_similarity(first_variables, second_variables)})

        first_functions = self.go_around_functions(first_ast)
        second_functions = self.go_around_functions(second_ast)
        params.update({'func_labels_similarity': self.calc_vocab_similarity(first_functions, second_functions)})

        params.update({
            'first_code_max_nesting': self.get_max_nesting(first_ast),
            'second_code_max_nesting': self.get_max_nesting(second_ast)
        })

        first_lexem_counter = self.get_lex_count(first_ast, (
        ast.For, ast.While, ast.ClassDef, ast.FunctionDef, ast.If, ast.Expression))
        second_lexem_counter = self.get_lex_count(second_ast, (
        ast.For, ast.While, ast.ClassDef, ast.FunctionDef, ast.If, ast.Expression))
        first_lexem_counter = defaultdict(lambda: 0, first_lexem_counter)
        second_lexem_counter = defaultdict(lambda: 0, second_lexem_counter)

        params.update(
            {
                "first_cycles_count": first_lexem_counter["<class 'ast.For'>"]
                                      + first_lexem_counter["<class 'ast.While'>"],
                "second_cycles_count": second_lexem_counter["<class 'ast.For'>"]
                                       + second_lexem_counter["<class 'ast.While'>"],
                "first_classes_count": first_lexem_counter["<class 'ast.ClassDef'>"],
                "second_classes_count": second_lexem_counter["<class 'ast.ClassDef'>"],
                "first_functions_count": first_lexem_counter["<class 'ast.FunctionDef'>"],
                "second_functions_count": second_lexem_counter["<class 'ast.FunctionDef'>"],
                "first_expressions_count": first_lexem_counter["<class 'ast.Expression'>"],
                "second_expressions_count": second_lexem_counter["<class 'ast.Expression'>"],
                "first_if_count": first_lexem_counter["<class 'ast.If'>"],
                "second_if_count": second_lexem_counter["<class 'ast.If'>"]
            }
        )

        first_code = self.normalize_code(first_code)
        second_code = self.normalize_code(second_code)

        return params

    def get_info_raw(self, first_code: str, second_code: str):
        parsed_pair = self.get_info(first_code, second_code)

        features = parsed_pair.items()
        features = sorted(features)
        column = []

        for name, value in features:
            column.append(value)

        return column