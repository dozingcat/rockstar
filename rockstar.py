# https://github.com/dylanbeattie/rockstar

from enum import Enum

class TokenError(Exception):
    pass

def tokenize(line: str, line_index=0):
    class State(Enum):
        BETWEEN_TOKENS = 1
        IN_WORD = 2
        IN_NUMBER = 3
        IN_QUOTE = 4

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
            elif ch in ("'", '"'):
                token_start = index
                quote_char = ch
                state = State.IN_QUOTE
            elif ch.isspace():
                pass
            else:
                raise TokenError(f'Unexpected character {ch} at line {line_index+1}:{index+1}')
        elif state == State.IN_WORD:
            if ch.isspace():
                tokens.append(line[token_start:index])
                state = State.BETWEEN_TOKENS
            else:
                pass
        elif state == State.IN_NUMBER:
            if ch.isspace():
                tokens.append(line[token_start:index])
                state = State.BETWEEN_TOKENS
            elif ch.isdigit() or ch in ('.', 'e', '+', '-'):
                pass
            else:
                raise TokenError(
                    f'Unexpected character {ch} in number at line {line_index+1}:{index+1}')
        elif state == State.IN_QUOTE:
            # TODO: Allow backslash escapes
            if ch == quote_char:
                tokens.append(line[token_start+1:index])
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


class Function:
    def __init__(self):
        pass

class Runtime:
    def __init__(self):
        pass
