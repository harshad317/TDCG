def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_iv = sorted([list(p) for p in intervals], key=lambda x: x[0])
    out = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
