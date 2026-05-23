def flatten(nested):
    out = []

    def go(x):
        if isinstance(x, list):
            for e in x:
                go(e)
        else:
            out.append(x)

    go(nested)
    return out
