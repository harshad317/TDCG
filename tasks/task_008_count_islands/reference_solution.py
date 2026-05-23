def count_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows = len(grid)
    cols = len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not seen[i][j]:
                count += 1
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1 and not seen[r][c]:
                        seen[r][c] = True
                        stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
    return count
