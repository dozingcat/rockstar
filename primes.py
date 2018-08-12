def mod(n, d):
    while n >= d:
        n -= d
    return n

def main():
    print(2)
    print(3)
    limit = 10000
    candidate = 5
    while candidate < limit:
        factor = 1
        prime = True
        while prime and factor * factor <= candidate:
            factor += 2
            if mod(candidate, factor) == 0:
                prime = False
        if prime:
            print(candidate)
        candidate += 2

if __name__ == '__main__':
    main()
