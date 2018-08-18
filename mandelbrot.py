# Reference implementation for mandelbrot.rock

def escape(real, imag):
    iters = 0
    r0, i0 = real, imag
    while iters < 100:
        real, imag = real*real - imag*imag + r0, 2*real*imag + i0
        if real*real + imag*imag > 4:
            break
        iters += 1
    return iters


def compute_lines(xmin, xmax, ymin, ymax, rows, cols):
    for r in range(rows):
        y = ymax - (ymax - ymin) * (r / rows)
        line = []
        for c in range(cols):
            x = xmin + (xmax - xmin) * (c / cols)
            iters = escape(x, y)
            line.append('.' if iters < 100 else '*')
        yield ''.join(line)


def main():
    for line in compute_lines(xmin=-2.0, xmax=1.0, ymin=-1.25, ymax=1.25, rows=32, cols=79):
        print(line)


if __name__ == '__main__':
    main()
