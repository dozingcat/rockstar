# rockstar
This is a Python implementation of the Rockstar programming language defined at
https://github.com/dylanbeattie/rockstar

It requires Python 3.6 or later. To execute a Rockstar program, run `rockstar.py [source file]`.

The Rockstar language definition may change frequently; the version targeted by this implementation is
https://github.com/dylanbeattie/rockstar/tree/864b14b4a40e5fd5cf372880c097c5472a52af1b

It can run both the 'minimal' and 'poetic' versions of the FizzBuzz sample code, but almost certainly has bugs.
Known limitations:
- "Else" blocks are not supported.
- The only allowed comments are entire lines beginning with '(' and ending with ')'.
This causes the line to be treated as blank, ending any active if/while/until block.
