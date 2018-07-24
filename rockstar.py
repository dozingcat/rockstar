#!/usr/bin/env python3

# Implementation of Rockstar language described at https://github.com/dylanbeattie/rockstar
# Current as of https://github.com/dylanbeattie/rockstar/commit/864b14b4a40e5fd5cf372880c097c5472a52af1b
#
# (c) 2018 Brian Nenninger
# Released under the MIT license.

from collections import namedtuple
from enum import Enum
import sys


# This is the value of the nothing/nowhere/nobody keyword.
# It acts like 0 and the empty string.
class _Null:
    def __eq__(self, other):
        return self is other or other == 0 or other == False or other == None

    def __hash__(self):
        return 0

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return -other

    def __rsub__(self, other):
        return -other

    def __mul__(self, other):
        return 0

    def __rmul__(self, other):
        return 0

    def __div__(self, other):
        return 0

    def __fdiv__(self, other):
        return 0

    def __gt__(self, other):
        return other > 0

    def __ge__(self, other):
        return other >= 0

    def __lt__(self, other):
        return other < 0

    def __le__(self, other):
        return other <= 0

    def __str__(self):
        return '[null]'


NULL = _Null()


class CompareOperator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    GT = 3
    GE = 4
    LT = 5
    LE = 6


class LogicalOperator(Enum):
    AND = 1
    OR = 2


class ArithmeticOperator(Enum):
    ADD = 1
    SUBTRACT = 2
    MULTIPLY = 3
    DIVIDE = 4

class BlockReturn(Exception):
    def __init__(self, value):
        self.value = value


class BlockBreak(Exception):
    pass


class BlockContinue(Exception):
    pass


class StackFrame:
    def __init__(self, fn_table, parent=None, stdin=sys.stdin, stdout=sys.stdout):
        self.parent = parent
        self.fn_table = fn_table
        self.vars = {}
        self.stdin = stdin
        self.stdout = stdout

    def get_var(self, var_name):
        if var_name not in self.vars:
            raise RuntimeError(f'Unknown variable: {var_name}')
        return self.vars[var_name]

    def set_var(self, var_name, value):
        self.vars[var_name] = value
        return value

    def add_function(self, func):
        self.fn_table[func.func_name] = func

    def call_function(self, fn_name, args):
        if fn_name not in self.fn_table:
            raise RuntimeError(f'Unknown function: ${fn_name}')
        fn = self.fn_table[fn_name]
        if len(args) != len(fn.parameters):
            raise RuntimeError(
                f'Expected {len(fn.parameters)} in call to {fn_name} but got {len(args)}')
        child_frame = StackFrame(self.fn_table, self, self.stdin, self.stdout)
        for param_name, param_value in zip(fn.parameters, args):
            child_frame.set_var(param_name, param_value)
        try:
            return fn.block.evaluate(child_frame)
        except BlockReturn as ret:
            return ret.value
        except BlockBreak:
            raise RuntimeError('break called from top level of function')
        except BlockContinue:
            raise RuntimeError('continue called from top level of function')


class Block(namedtuple('Block', ['expressions'])):
    def evaluate(self, frame: StackFrame):
        lastval = None
        for expr in self.expressions:
            lastval = expr.evaluate(frame)
        return lastval


class FunctionExpression(namedtuple('Function', ['func_name', 'parameters', 'block'])):
    def evaluate(self, frame: StackFrame):
        frame.add_function(self)


class ConstantExpression(namedtuple('ConstantExpression', ['value'])):
    def evaluate(self, frame: StackFrame):
        return self.value


class VariableExpression(namedtuple('AssignToVarStatement', ['var_name'])):
    def evaluate(self, frame: StackFrame):
        return frame.get_var(self.var_name)


class CompareExpression(namedtuple('CompareExpression', ['left_expr', 'operator', 'right_expr'])):
    def evaluate(self, frame: StackFrame):
        lhs = self.left_expr.evaluate(frame)
        rhs = self.right_expr.evaluate(frame)
        if self.operator == CompareOperator.EQUAL:
            return lhs == rhs
        if self.operator == CompareOperator.NOT_EQUAL:
            return lhs != rhs
        if self.operator == CompareOperator.GT:
            return lhs > rhs
        if self.operator == CompareOperator.GE:
            return lhs >= rhs
        if self.operator == CompareOperator.LT:
            return lhs < rhs
        if self.operator == CompareOperator.LE:
            return lhs <= rhs
        raise AssertionError(f'Unknown operator: {self.operator}')


class ArithmeticExpression(namedtuple('ArithmeticExpression',
                                      ['left_expr', 'operator', 'right_expr'])):
    def evaluate(self, frame: StackFrame):
        lhs = self.left_expr.evaluate(frame)
        rhs = self.right_expr.evaluate(frame)
        if self.operator == ArithmeticOperator.ADD:
            return lhs + rhs
        if self.operator == ArithmeticOperator.SUBTRACT:
            return lhs - rhs
        if self.operator == ArithmeticOperator.MULTIPLY:
            return lhs * rhs
        if self.operator == ArithmeticOperator.DIVIDE:
            return lhs / rhs
        raise AssertionError(f'Unknown operator: {self.operator}')


class LogicalExpression(namedtuple('CompareExpression', ['left_expr', 'operator', 'right_expr'])):
    def evaluate(self, frame: StackFrame):
        lhs = self.left_expr.evaluate(frame)
        # short circuit
        if self.operator == LogicalOperator.AND:
            return bool(lhs and self.right_expr.evaluate(frame))
        if self.operator == LogicalOperator.OR:
            return bool(lhs or self.right_expr.evaluate(frame))
        raise AssertionError(f'Unknown operator: {self.operator}')


class NegateBinaryExpression(namedtuple('NegateBinaryExpression', ['expr'])):
    def evaluate(self, frame: StackFrame):
        return not self.expr.evaluate(frame)


class AssignmentExpression(namedtuple('AssignmentExpression', ['var_name', 'expr'])):
    def evaluate(self, frame: StackFrame):
        return frame.set_var(self.var_name, self.expr.evaluate(frame))


class AddToVarExpression(namedtuple('AddToVarExpression', ['target_var', 'expr'])):
    def evaluate(self, frame: StackFrame):
        return frame.set_var(self.target_var,
                             frame.get_var(self.target_var) + self.expr.evaluate(frame))


class SubtractFromVarExpression(namedtuple('SubtractFromVarExpression', ['target_var', 'src_var'])):
    def evaluate(self, frame: StackFrame):
        return frame.set_var(frame.get_var(self.target_var) - self.expr.evaluate(frame))


class CallFunctionExpression(namedtuple('CallFunctionExpression', ['func_name', 'args'])):
    def evaluate(self, frame: StackFrame):
        return frame.call_function(self.func_name, [arg.evaluate(frame) for arg in self.args])


class WhileExpression(namedtuple('WhileExpression', ['expr', 'block'])):
    def evaluate(self, frame: StackFrame):
        while self.expr.evaluate(frame):
            try:
                self.block.evaluate(frame)
            except BlockBreak:
                return
            except BlockContinue:
                continue


class IfExpression(namedtuple('IfExpression', ['expr', 'true_block', 'false_block'])):
    def evaluate(self, frame: StackFrame):
        block = self.true_block if self.expr.evaluate(frame) else self.false_block
        block.evaluate(frame)


class ReturnExpression(namedtuple('ReturnExpression', ['expr'])):
    def evaluate(self, frame: StackFrame):
        raise BlockReturn(self.expr.evaluate(frame))


class BreakExpression:
    def evaluate(self, frame: StackFrame):
        raise BlockBreak


class ContinueExpression:
    def evaluate(self, frame: StackFrame):
        raise BlockContinue


class WriteExpression(namedtuple('WriteExpression', ['expr'])):
    def evaluate(self, frame: StackFrame):
        val = self.expr.evaluate(frame)
        print(val, file=frame.stdout)
        return val


class ReadExpression(namedtuple('ReadExpression', ['var_name'])):
    def evaluate(self, frame: StackFrame):
        line = frame.stdin.readline()
        frame.set_var(self.var_name, line)
        return line


class TokenError(Exception):
    pass


class ParseError(Exception):
    pass


class ParseContext:
    def __init__(self, is_condition=False):
        self.last_var = None
        self.is_condition = False
        self.debug = False


ASSIGNMENT_KEYWORDS = ['is', 'was', 'were', 'says']
STRING_ASSIGNMENT_KEYWORDS = 'says'
COMMON_VARIABLE_PREFIXES = ['a', 'an', 'the', 'my', 'your']
PRONOUNS = ['it', 'he', 'she', 'him', 'her', 'them', 'they']
QUOTE_CHARS = ['"']
SEPARATOR_CHARS = [',']
POETIC_LITERALS = {
    'true': True,
    'false': False,
    'nothing': NULL,
    'nobody': NULL,
    'nowhere': NULL,
}
INCREMENT_KEYWORDS = {
    1: ['build', 'up'],
    -1: ['knock', 'down'],
}
ARITHMETIC_KEYWORDS = {
    ArithmeticOperator.ADD: ['plus', 'with'],
    ArithmeticOperator.SUBTRACT: ['minus', 'without'],
    ArithmeticOperator.MULTIPLY: ['times', 'of'],
    ArithmeticOperator.DIVIDE: ['over', 'by'],
}
# Compare operators can be multi-word, e.g. "is not", "is bigger than".
COMPARE_OPERATORS = {
    CompareOperator.GT: [
        ['is', 'higher', 'than'],
        ['is', 'greater', 'than'],
        ['is', 'bigger', 'than'],
        ['is', 'stronger', 'than'],
        ['is', 'not', 'as', 'low', 'as'],
        ['is', 'not', 'as', 'little', 'as'],
        ['is', 'not', 'as', 'small', 'as'],
        ['is', 'not', 'as', 'weak', 'as'],
    ],
    CompareOperator.GE: [
        ['is', 'as', 'high', 'as'],
        ['is', 'as', 'great', 'as'],
        ['is', 'as', 'big', 'as'],
        ['is', 'as', 'strong', 'as'],
        ['is', 'not', 'lower', 'than'],
        ['is', 'not', 'less', 'than'],
        ['is', 'not', 'smaller', 'than'],
        ['is', 'not', 'weaker', 'than'],
    ],
    CompareOperator.LT: [
        ['is', 'lower', 'than'],
        ['is', 'less', 'than'],
        ['is', 'smaller', 'than'],
        ['is', 'weaker', 'than'],
        ['is', 'not', 'as', 'high', 'as'],
        ['is', 'not', 'as', 'great', 'as'],
        ['is', 'not', 'as', 'big', 'as'],
        ['is', 'not', 'as', 'strong', 'as'],
    ],
    CompareOperator.LE: [
        ['is', 'as', 'low', 'as'],
        ['is', 'as', 'little', 'as'],
        ['is', 'as', 'small', 'as'],
        ['is', 'as', 'weak', 'as'],
        ['is', 'not', 'higher', 'than'],
        ['is', 'not', 'greater', 'than'],
        ['is', 'not', 'bigger', 'than'],
        ['is', 'not', 'stronger', 'than'],
    ],
    CompareOperator.NOT_EQUAL: [['is', 'not'], ["ain't"]],
    CompareOperator.EQUAL: [['is']],
}
LOGICAL_KEYWORDS = {
    LogicalOperator.AND: ['and'],
    LogicalOperator.OR: ['or'],
}
BREAK_KEYWORDS = [['break'], ['break', 'it', 'down']]
CONTINUE_KEYWORDS = [['continue'], ['take', 'it', 'to', 'the', 'top']]
RETURN_KEYWORDS = [['give', 'back']]
FUNCTION_CALL_KEYWORDS = ['taking']
WRITE_KEYWORDS = [['say'], ['shout'], ['whisper'], ['scream']]
READ_KEYWORDS = [['listen to']]
CONDITION_KEYWORDS = ['if', 'while', 'until']


def variable_name(tokens, context=None):
    def record(var_name):
        if context is not None:
            context.last_var = var_name
        return var_name

    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].lower() in PRONOUNS:
        if context is None or context.last_var is None:
            if not context.last_var:
                raise ValueError(f'Pronoun {val} has no antecedent')
        return context.last_var
    if tokens[0].lower() in COMMON_VARIABLE_PREFIXES:
        if len(tokens) == 2 and tokens[1].isalpha() and tokens[1].islower():
            return record(' '.join(tokens).lower())
    if all(t[0].isupper() for t in tokens):
        return record(' '.join(tokens))

def is_poetic_prefix(tokens):
    if len(tokens) < 2:
        return False
    if tokens[-1].lower() not in ASSIGNMENT_KEYWORDS:
        return False
    if tokens[0].lower() in CONDITION_KEYWORDS:
        return False
    return variable_name(tokens[:-1]) is not None


def tokenize(line: str, line_index=0):
    class State(Enum):
        BETWEEN_TOKENS = 1
        IN_WORD = 2
        IN_NUMBER = 3
        IN_QUOTE = 4
        AFTER_POETIC_ASSIGNMENT = 5

    tokens = []
    token_start = -1
    quote_char = ''
    state = State.BETWEEN_TOKENS
    for index, ch in enumerate(line):
        if state == State.BETWEEN_TOKENS:
            if ch.isalpha():
                token_start = index
                state = State.IN_WORD
            elif ch.isdigit():
                token_start = index
                state = State.IN_NUMBER
            elif ch in QUOTE_CHARS:
                token_start = index
                quote_char = ch
                state = State.IN_QUOTE
            elif ch in SEPARATOR_CHARS:
                tokens.append(ch)
            elif ch.isspace():
                pass
            else:
                raise TokenError(f'Unexpected character {ch} at line {line_index+1}:{index+1}')
        elif state == State.IN_WORD:
            if ch.isspace():
                tokens.append(line[token_start:index])
                if is_poetic_prefix(tokens):
                    poetic_val = line[index+1:].strip()
                    if not poetic_val:
                        raise TokenError(f'Expected poetic value at line {line_index+1}:{index+1}')
                    tokens.append(poetic_val)
                    state = State.AFTER_POETIC_ASSIGNMENT
                    break
                else:
                    state = State.BETWEEN_TOKENS
            elif ch in SEPARATOR_CHARS:
                tokens.append(line[token_start:index])
                tokens.append(ch)
                state = State.BETWEEN_TOKENS
            else:
                pass
        elif state == State.IN_NUMBER:
            if ch.isspace():
                tokens.append(line[token_start:index])
                state = State.BETWEEN_TOKENS
            elif ch.isdigit() or ch in ('.', 'e', '+', '-'):
                pass
            elif ch in SEPARATOR_CHARS:
                tokens.append(line[token_start:index])
                tokens.append(ch)
                state = State.BETWEEN_TOKENS
            else:
                raise TokenError(
                    f'Unexpected character {ch} in number at line {line_index+1}:{index+1}')
        elif state == State.IN_QUOTE:
            # Do we need escapes?
            if ch == quote_char:
                # Include the quotes in the token so we can distinguish "123" from 123.
                tokens.append(line[token_start:index+1])
                state = State.BETWEEN_TOKENS
            else:
                pass
        else:
            raise AssertionError(f'Unknown state: {state}')
    # Handle end of line.
    if state == State.IN_WORD or state == State.IN_NUMBER:
        tokens.append(line[token_start:])
    elif state == State.IN_QUOTE:
        raise TokenError(f'Unterminated quote starting at line {line_index+1}:{index+1}')
    return tokens


def parse_possibly_poetic(val: str):
    if val in POETIC_LITERALS:
        return POETIC_LITERALS[val]
    # int or float literal
    if val[0].isdigit():
        return float(val) if '.' in val else int(val)
    # poetic number literal. If there's a period, handle left and right.
    def to_digits(s):
        words = s.split()
        return ''.join(str(sum(ch.isalpha() for ch in word) % 10) for word in words)

    if '.' in val:
        left, right = val.split('.', maxsplit=1)
        return float(to_digits(left) + '.' + to_digits(right))
    else:
        return int(to_digits(val))


def _find_sublist(items, sublist):
    lsub = len(sublist)
    for i in range(len(items) - lsub + 1):
        if sublist == items[i:i+lsub]:
            return i
    return None


def _lists_equal_ignoring_case(list1, list2):
    return len(list1) == len(list2) and [s.upper() for s in list1] == [s.upper() for s in list2]


def parse_expression(tokens, context):
    context = context or ParseContext()
    # Strip trailing commas.
    while len(tokens) > 0 and tokens[-1] in SEPARATOR_CHARS:
        tokens.pop()
    # Literal value or pronoun.
    if len(tokens) == 1:
        val = tokens[0]
        if val in POETIC_LITERALS:
            return ConstantExpression(POETIC_LITERALS[val])
        if val[0] in QUOTE_CHARS:
            return ConstantExpression(val[1:-1])
        if all(ch in '+-.e' or ch.isdigit() for ch in val):
            return ConstantExpression(float(val) if '.' in val else int(val))

    # Assignment: put [expr] into [var_name]
    if len(tokens) >= 4 and tokens[0].lower() == 'put' and 'into' in tokens:
        into_index = next(i for i in range(len(tokens) - 1, -1, -1) if tokens[i] == 'into')
        if into_index < 2:
            raise ValueError('Missing value between "put" and "into"')
        var_name = variable_name(tokens[into_index+1:], context)
        if not var_name:
            raise ValueError('Variable name not found after "into"')
        expr = parse_expression(tokens[1:into_index], context)
        return AssignmentExpression(var_name, expr)

    # Logical operators
    if len(tokens) >= 3:
        for operator, keywords in LOGICAL_KEYWORDS.items():
            for kw in keywords:
                if kw in tokens:
                    kw_index = tokens.index(kw)
                    if kw_index > 0 and kw_index < len(tokens) - 1:
                        left_expr = parse_expression(tokens[:kw_index], context)
                        right_expr = parse_expression(tokens[kw_index+1:], context)
                        return LogicalExpression(left_expr, operator, right_expr)

    # Poetic assignment: [var_name] [is/was/says...] [value]. Don't do this for
    # the target condition of an if/while statement, because "Foo is 4" is an
    # assignment but "If Foo is 4" is a compare operation.
    if not context.is_condition and len(tokens) >= 3 and tokens[-2].lower() in ASSIGNMENT_KEYWORDS:
        var_name = variable_name(tokens[:-2], context)
        if var_name:
            # 'Alice says ZZZ' is always a string assignment
            if tokens[-2] in STRING_ASSIGNMENT_KEYWORDS:
                return AssignmentExpression(var_name, ConstantExpression(tokens[-1]))
            value = parse_possibly_poetic(tokens[-1])
            return AssignmentExpression(var_name, ConstantExpression(value))

    # Increment/decrement
    if len(tokens) >= 3:
        for amount, keywords in INCREMENT_KEYWORDS.items():
            if tokens[0].lower() == keywords[0] and tokens[-1].lower() == keywords[1]:
                var_name = variable_name(tokens[1:-1], context)
                if var_name:
                    return AddToVarExpression(var_name, ConstantExpression(amount))

    # Compare operators
    if len(tokens) >= 3:
        for operator, kw_lists in COMPARE_OPERATORS.items():
            for kw_list in kw_lists:
                kw_index = _find_sublist(tokens, kw_list)
                if kw_index is not None:
                    left_expr = parse_expression(tokens[:kw_index], context)
                    right_expr = parse_expression(tokens[kw_index + len(kw_list):], context)
                    return CompareExpression(left_expr, operator, right_expr)

    # Arithmetic
    if len(tokens) >= 3:
        for operator, keywords in ARITHMETIC_KEYWORDS.items():
            for kw in keywords:
                if kw in tokens:
                    kw_index = tokens.index(kw)
                    if kw_index > 0 and kw_index < len(tokens) - 1:
                        left_expr = parse_expression(tokens[:kw_index], context)
                        right_expr = parse_expression(tokens[kw_index+1:], context)
                        return ArithmeticExpression(left_expr, operator, right_expr)

    # Break/continue
    ltokens = [t.lower() for t in tokens]
    if any(kw_list == ltokens for kw_list in BREAK_KEYWORDS):
        return BreakExpression()
    if any(kw_list == ltokens for kw_list in CONTINUE_KEYWORDS):
        return ContinueExpression()

    # Return
    for kw_list in RETURN_KEYWORDS:
        if ltokens[:len(kw_list)] == kw_list:
            ret_expr = parse_expression(tokens[len(kw_list):], context)
            return ReturnExpression(ret_expr)

    # Function call
    for kw in FUNCTION_CALL_KEYWORDS:
        if kw in tokens:
            kw_index = tokens.index(kw)
            if kw_index > 0:
                fn_name = variable_name(tokens[:kw_index], context)
                arg_exprs = []
                current_arg_tokens = []
                for t in tokens[kw_index+1:]:
                    if t in SEPARATOR_CHARS:
                        arg_exprs.append(parse_expression(current_arg_tokens, context))
                        current_arg_tokens = []
                    else:
                        current_arg_tokens.append(t)
                if len(current_arg_tokens) > 0:
                    arg_exprs.append(parse_expression(current_arg_tokens, context))
                return CallFunctionExpression(fn_name, arg_exprs)

    # Output
    for kw_list in WRITE_KEYWORDS:
        if ltokens[:len(kw_list)] == kw_list:
            expr = parse_expression(tokens[len(kw_list):], context)
            return WriteExpression(expr)

    # Input
    for kw_list in READ_KEYWORDS:
        if ltokens[:len(kw_list)] == kw_list:
            var_name = variable_name(tokens[len(kw_list):], context)
            return ReadExpression(var_name)

    # Variable name
    var_name = variable_name(tokens, context)
    if var_name:
        return VariableExpression(var_name)

    raise ParseError('Failed to parse tokens: ', tokens)


class BlockType(Enum):
    ROOT = 0
    IF = 1
    ELSE = 2
    WHILE = 3
    UNTIL = 4
    FUNCTION = 5


class BlockBuilder:
    def __init__(self, btype, create_expr_fn):
        self.type = btype
        self.subexpressions = []
        self.create_block_expr = lambda: create_expr_fn(Block(self.subexpressions))


def parse_lines(lines, debug=False):
    block_stack = [BlockBuilder(BlockType.ROOT, lambda block: None)]
    context = ParseContext()

    def end_current_block():
        if len(block_stack) <= 1:
            raise RuntimeError('No block to end')
        ending_block = block_stack.pop()
        block_expr = ending_block.create_block_expr()
        if (debug):
            print('Adding block expression: ', ending_block.type, block_expr)
        block_stack[-1].subexpressions.append(block_expr)


    for line_index, line in enumerate(lines):
        try:
            line = line.strip()
            # TODO: allow multiline and within a line (but make sure they're not in quotes, etc).
            if line.startswith('(') and line.endswith(')'):
                line = ''
            tokens = tokenize(line, line_index)
            if len(tokens) == 0:
                # Block ends
                if len(block_stack) > 1:
                    end_current_block()
                continue
            # Does case matter for these keywords?
            first = tokens[0].lower()
            if first == 'if':
                # Nested function so `condition` doesn't get reassigned.
                def doit():
                    context.is_condition = True
                    condition = parse_expression(tokens[1:], context)
                    block_stack.append(BlockBuilder(
                        BlockType.IF,
                        lambda block: IfExpression(condition, block, Block([]))))
                doit()
            elif first == 'while':
                def doit():
                    context.is_condition = True
                    while_condition = parse_expression(tokens[1:], context)
                    block_stack.append(BlockBuilder(
                        BlockType.WHILE, lambda block: WhileExpression(while_condition, block)))
                doit()
            elif first == 'until':
                # "Until Foo" is equivalent to "While not Foo"
                def doit():
                    context.is_condition = True
                    until_condition = NegateBinaryExpression(parse_expression(tokens[1:], context))
                    block_stack.append(BlockBuilder(
                        BlockType.UNTIL, lambda block: WhileExpression(until_condition, block)))
                doit()
            elif 'takes' in tokens:
                def doit():
                    tindex = tokens.index('takes')
                    fn_name = variable_name(tokens[:tindex])
                    param_names = []
                    param_name_tokens = []
                    for t in tokens[tindex+1:]:
                        if t == 'and':
                            param_names.append(variable_name(param_name_tokens, context))
                            param_name_tokens = []
                        else:
                            param_name_tokens.append(t)
                    if len(param_name_tokens) > 0:
                        param_names.append(variable_name(param_name_tokens, context))
                    block_stack.append(BlockBuilder(
                        BlockType.FUNCTION,
                        lambda block: FunctionExpression(fn_name, param_names, block)))
                doit()
            # TODO: else
            else:
                context.is_condition = False
                expr = parse_expression(tokens, context)
                block_stack[-1].subexpressions.append(expr)
        except Exception as ex:
            print(f'Error at line {line_index+1}')
            raise

    while len(block_stack) > 1:
        end_current_block()
    return Block(block_stack[0].subexpressions)


def execute_lines(lines, debug=False, stdin=sys.stdin, stdout=sys.stdout):
    frame = StackFrame({}, stdin=stdin, stdout=stdout)
    block = parse_lines(lines, debug)
    if debug:
        print('Expressions: ', block.expressions)
        print('*** Starting execution ***')
    block.evaluate(frame)
    if debug:
        print('*** Execution finished ***')
        print('variables:', frame.vars)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} [file to run]', file=sys.stderr)
        sys.exit(1)
    infile = sys.argv[1]
    with open(infile) as f:
        lines = f.readlines()
    execute_lines(lines, debug=False)
