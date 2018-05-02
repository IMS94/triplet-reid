def compare(v1, v2):
    total_squared_d = 0
    for i in range(len(v1)):
        total_squared_d += (v1[i] - v2[i]) ** 2
    return total_squared_d ** 0.5
