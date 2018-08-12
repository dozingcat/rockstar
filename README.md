# rockstar
This is a Python implementation of the Rockstar programming language defined at
https://github.com/dylanbeattie/rockstar

It requires Python 3.6 or later. (Although it appears to work with pypy 3.5 as well). To execute a Rockstar program, run `rockstar.py [source file]`.

The Rockstar language definition may change frequently; the version targeted by this implementation is
https://github.com/dylanbeattie/rockstar/tree/699d9d87992ecbb17d34ad010e2cdd6df8d71791

It can run both the 'minimal' and 'poetic' versions of the FizzBuzz sample code, but almost certainly has bugs.
Known limitations:
- "Else" blocks are not supported.
- The only allowed comments are entire lines beginning with '(' and ending with ')'.
This causes the line to be treated as blank, ending any active if/while/until block.
