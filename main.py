import timeit
import pprint
keys = ['1', '2', '3','4','5','6','7','8','9','10','11','12']
value = {"assignment": "ok"}
valueB = {"increment": 100}

def double_loop():
    parent_lib = {}
    for i in keys:
        parent_lib[i] = {}
    for i in keys:
        for j in range(0, 100):
            new_key = f"key_{j}"
            parent_lib[i][new_key] = value


def single_loop_with_check():
    another_lib = {}
    for i in keys:
        another_lib.setdefault(i, {})["new"] = value
        another_lib.setdefault(i, {})["newB"] = valueB

# Run each function 100,000 times to get an accurate average
print("Double Loop:", timeit.timeit(double_loop, number=1000))
print("Single Loop:", timeit.timeit(single_loop_with_check, number=1000))
