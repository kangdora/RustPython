import sys

from testutils import assert_raises

N = 20000
HOT = 5000


def while_sum(n):
    i = 0
    total = 0
    while i < n:
        total += i
        i += 1
    return total


def for_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total


def nested(n):
    acc = 0
    for i in range(n):
        j = 0
        while j < 3:
            acc += i * j
            j += 1
    return acc


def with_break(n):
    i = 0
    while True:
        if i >= n:
            break
        i += 1
    return i


def calls_inside(n):
    total = 0
    for i in range(n):
        total += abs(-i)
    return total


def raises_inside(n):
    i = 0
    try:
        while True:
            i += 1
            if i == n:
                1 / 0
    except ZeroDivisionError:
        return i


def raises_out(n):
    i = 0
    while True:
        i += 1
        if i == n:
            raise ValueError(i)


def unbound_local(n):
    i = 0
    while i < n:
        i += 1
        if i == n:
            return later
    later = 1
    return later


def none_walk(n):
    node = n
    steps = 0
    while node is not None:
        steps += 1
        node = node - 1 if node > 0 else None
    return steps


def float_and_not(n):
    x = 0.0
    flag = True
    while not flag or x < n:
        x += 0.5
        flag = not flag
    return x, flag


def negate(n):
    acc = 0
    for i in range(n):
        acc += -i
    return acc


def list_walk(items):
    total = 0
    for x in items:
        total += x
    return total


def gen(n):
    i = 0
    while i < n:
        yield i
        i += 1


def compare_chain(n):
    hits = 0
    for i in range(n):
        if 10 <= i < 20:
            hits += 1
    return hits


def globals_in_loop(n):
    total = 0
    for i in range(n):
        total += HOT
    return total


def str_concat(n):
    s = ""
    for i in range(n):
        s += "a"
    return len(s)


def check_all():
    assert while_sum(N) == N * (N - 1) // 2
    assert for_sum(N) == N * (N - 1) // 2
    assert nested(HOT) == 3 * HOT * (HOT - 1) // 2
    assert with_break(N) == N
    assert calls_inside(N) == N * (N - 1) // 2
    assert raises_inside(N) == N
    with assert_raises(ValueError):
        raises_out(N)
    with assert_raises(UnboundLocalError):
        unbound_local(N)
    assert none_walk(N) == N + 1
    assert float_and_not(N) == (N, True)
    assert negate(N) == -(N * (N - 1) // 2)
    assert list_walk(list(range(N))) == N * (N - 1) // 2
    assert sum(gen(N)) == N * (N - 1) // 2
    assert compare_chain(N) == 10
    assert globals_in_loop(N) == N * HOT
    assert str_concat(N) == N


check_all()
check_all()

jit = getattr(sys, "_jit", None)
if jit is not None and jit.is_enabled():
    stats = jit.stats()
    assert stats["compiled"] >= 1, stats
    assert stats["entered"] >= 1, stats
    before = jit.stats()["entered"]
    assert while_sum(N) == N * (N - 1) // 2
    assert jit.stats()["entered"] > before
    before = jit.stats()["errors"]
    with assert_raises(UnboundLocalError):
        unbound_local(N)
    assert jit.stats()["errors"] > before
