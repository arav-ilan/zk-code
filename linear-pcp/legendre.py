import math
import random as rand
from typing_extensions import Optional

ROBUST_TEST_SET: list[tuple[int, list[int]]] = [
  (2047,                  [ 2 ]),
  (1373653,               [ 2, 3 ]),
  (9080191,               [ 31, 73 ]),
  (25326001,              [ 2, 3, 5 ]),
  (3215031751,            [ 2, 3, 5, 7 ]),
  (4759123141,            [ 2, 7, 61 ]),
  (1122004669633,         [ 2, 13, 23, 1662803 ]),
  (2152302898747,         [ 2, 3, 5, 7, 11 ]),
  (3474749660383,         [ 2, 3, 5, 7, 11, 13 ]),
  (341550071728321,       [ 2, 3, 5, 7, 11, 13, 17 ]),
  (3825123056546413051,   [ 2, 3, 5, 7, 11, 13, 17, 19, 23 ]),
  (18446744073709551616,  [ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 ]),
]

def is_perfect_square(x: int) -> bool:
  return int(math.sqrt(x)) ** 2 == x

# miller-robin prime probabilistic test
def miller_robin_test(n: int, a: int) -> bool:
  d = n - 1
  while d % 2 == 0:
    d //= 2
  
  x = pow(a, d, n)

  if x == 1 or x == n - 1:
    return True

  while d != n - 1:
    x = pow(x, 2, n)
    d *= 2
    if x == n - 1:
      return True
  
  return False

def is_prime(n: int) -> bool:
  if n == 2:
    return True
  if n <= 1 or n % 2 == 0:
    return False
  
  found = -1
  for idx, set in enumerate(ROBUST_TEST_SET):
    if set[0] >= n:
      found = idx
      break
  
  if (found > -1):
    tests = [ not miller_robin_test(n, a) for a in ROBUST_TEST_SET[found][1]]
    return sum(tests) == 0
  else:
    for _ in range(15):
      if not miller_robin_test(n, rand.randint(2, 2**128-1)):
        return False
    return True
  
def get_remainders(a: int, b: int, bound: int) -> list[int]:
  result = []
  bound_sqrt = int(math.sqrt(bound))
  while b > 0 and len(result) < 2:
    r = a - ( a // b ) * b
    if r > 0 and r <= bound_sqrt:
      result.append(r)
    a = b
    b = r
  return result

def check_rabin_exceptions(n: int) -> Optional[list[int]]:
  if n > 9634:
    return None
  else:
    match n:
      case 5: return [2,1,0]
      case 10: return [3,1,0]
      case 13: return [3,2,0]
      case 34: return [5,3,0]
      case 37: return [6,1,0]
      case 58: return [7,3,0]
      case 61: return [6,5,0]
      case 85: return [9,2,0]
      case 130: return [11,3,0]
      case 214: return [14,3,3]
      case 226: return [15,1,0]
      case 370: return [19,3,0]
      case 526: return [21,9,2]
      case 706: return [25,9,0]
      case 730: return [27,1,0]
      case 829: return [27,10,0]
      case 1414: return [33,18,1]
      case 1549: return [35,18,0]
      case 1906: return [41,15,0]
      case 2986: return [45,31,0]
      case 7549: return [85,18,0]
      case 9634: return [97,15,0]
      case _: return None

def sum_of_two(p: int) -> list[int]:
  if p == 1:
    return [0, 1]
  elif p == 2:
    return [1, 1]
  
  p_dec = p - 1
  p_dec_half = p_dec // 2
  p_dec_quarter = p_dec // 4
  
  while True:
    q = rand.randint(0, p - 1)
    while pow(q, p_dec_half, p) != p_dec:
      q = rand.randint(0, p - 1)
    x = pow(q, p_dec_quarter, p)
    rems = get_remainders(p, x, p)
    if len(rems) >= 2:
      return [rems[0], rems[1]]

def legendre(n: int) -> list[int]:
  if check_rabin_exceptions(n) is not None:
    return check_rabin_exceptions(n)
  elif n % 4 == 0:
    sub = legendre(n // 4)
    return [ sub[0]*2, sub[1]*2, sub[2]*2 ]
  elif is_perfect_square(n):
    return [ int(math.sqrt(n)), 0, 0]
  
  match (n - (n // 8) * 8):
    case 7:
      raise Exception("Legendre not exists")
    case 3:
      while True:
        x = rand.randint(0, int(math.sqrt(n)) + 1)
        p = (n - x**2) // 2
        if (n - x**2) % 2 == 0 and (is_prime(p) or p == 1):
          break
      two = sum_of_two(p)
      return [ x, two[0] + two[1], abs(two[0] - two[1])]
    case _:
      while True:
        x = rand.randint(0, int(math.sqrt(n)) + 1)
        p = n - x**2
        if is_prime(p):
          break
      two = sum_of_two(p)
      return [ x, two[0], two[1] ]
  

if __name__ == "__main__":
  import time
  
  # Correctness test
  print("Running correctness tests...")
  for i in range(100):
    while True:
      n = rand.randint(5, 2**64-1)
      if n % 4 == 1:
        break
    
    triple = legendre(n)
    assert sum([ x**2 for x in triple]) == n
    if (i + 1) % 20 == 0:
      print(f"  Passed {i + 1}/100 tests")
  
  print("\nAll correctness tests passed!")
  
  # Performance benchmark
  print("\nRunning performance benchmark...")
  
  test_sizes = [
    ("Small (< 2^16)", 2**8, 2**16, 1000),
    ("Medium (< 2^32)", 2**24, 2**32, 100),
    ("Large (< 2^64)", 2**48, 2**64-1, 10)
  ]
  
  for size_name, min_val, max_val, iterations in test_sizes:
    times = []
    
    for _ in range(iterations):
      while True:
        n = rand.randint(min_val, max_val)
        if n % 4 == 1:
          break
      
      start = time.perf_counter()
      triple = legendre(n)
      end = time.perf_counter()
      
      times.append(end - start)
      assert sum([x**2 for x in triple]) == n
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n{size_name}:")
    print(f"  Iterations: {iterations}")
    print(f"  Average time: {avg_time*1000:.3f} ms")
    print(f"  Min time: {min_time*1000:.3f} ms")
    print(f"  Max time: {max_time*1000:.3f} ms")