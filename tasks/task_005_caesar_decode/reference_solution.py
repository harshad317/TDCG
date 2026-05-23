def caesar_decode(s, shift):
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") - shift) % 26 + ord("a")))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - ord("A") - shift) % 26 + ord("A")))
        else:
            out.append(ch)
    return "".join(out)
