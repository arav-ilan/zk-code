# Fully Linear PCP — Section 4.2 / Figure 4 demo
# Field arithmetic over F_p
from random import SystemRandom
rng = SystemRandom()

def modinv(a, p):
    return pow(a % p, p - 2, p)  # p is prime

def poly_trim(c):
    while c and c[-1] == 0:
        c.pop()
    return c or [0]

def poly_add(a, b, p):
    m = max(len(a), len(b))
    out = [(0) for _ in range(m)]
    for i in range(m):
        out[i] = ((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p
    return poly_trim(out)

def poly_scalar_mul(a, s, p):
    return poly_trim([(x * s) % p for x in a])

def poly_mul(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % p
    return poly_trim(out)

def poly_eval(c, x, p):
    acc = 0
    for coeff in reversed(c):
        acc = (acc * x + coeff) % p
    return acc

def lagrange_basis_at(points, r, p):
    # points = [x0, x1, ..., xM] (distinct)
    lambdas = []
    for i, xi in enumerate(points):
        num, den = 1, 1
        for j, xj in enumerate(points):
            if j == i: continue
            num = (num * (r - xj)) % p
            den = (den * (xi - xj)) % p
        lambdas.append((num * modinv(den, p)) % p)
    return lambdas  # λ_i(r)

def lagrange_poly(points, ys, p):
    # Return coefficients of the unique degree-≤M polynomial f with f(points[i]) = ys[i]
    # Lagrange form: f(X) = sum_i ys[i]*L_i(X)
    # Build each L_i(X) explicitly
    out = [0]
    for i, xi in enumerate(points):
        # Build numerator ∏_{j≠i} (X - xj)
        numer = [1]
        den = 1
        for j, xj in enumerate(points):
            if j == i: continue
            numer = poly_mul(numer, [-xj % p, 1], p)  # (X - xj)
            den = (den * (xi - xj)) % p
        Li = poly_scalar_mul(numer, modinv(den, p), p)
        out = poly_add(out, poly_scalar_mul(Li, ys[i] % p, p), p)
    return poly_trim(out)

# Gate G(u, v, w) = u*v + w   (deg G = 2)
def G(u, v, w, p):
    return (u * v + w) % p

def prover_fig4(x1, x2, x3, w, p, strong_hvzk=True, z=None):
    M, L = 2, 3  # two G gates; arity three
    # Randomize constant terms (for strong HVZK)
    if z is None:
        if strong_hvzk:
            z1, z2, z3 = (rng.randrange(0, p) for _ in range(3))
        else:
            z1 = z2 = z3 = 0
    else:
        z1, z2, z3 = z

    # Values required by Fig. 4
    v1 = G(x1, x2, x3, p)
    # Interpolation points X = {0,1,2}
    Xs = [0, 1, 2]
    f1_vals = [z1, x1, v1]
    f2_vals = [z2, x2, x3]
    f3_vals = [z3, x3, (x3 + w + 7) % p]

    # Build polynomials f1, f2, f3 of degree ≤ M by Lagrange interpolation
    f1 = lagrange_poly(Xs, f1_vals, p)
    f2 = lagrange_poly(Xs, f2_vals, p)
    f3 = lagrange_poly(Xs, f3_vals, p)

    # p(X) = G(f1(X), f2(X), f3(X)) = f1*f2 + f3
    p_poly = poly_add(poly_mul(f1, f2, p), f3, p)

    proof = {
        "w": w % p,
        "z": (z1 % p, z2 % p, z3 % p),
        "cp": p_poly,          # coefficients (constant term first)
        # debug/explanatory (not part of the FLPCP proof vector):
        "_f1": f1, "_f2": f2, "_f3": f3
    }
    return proof

def verifier_fig4(x1, x2, x3, proof, p, r=None):
    # Extract proof elements
    w = proof["w"] % p
    z1, z2, z3 = proof["z"]
    cp = proof["cp"]

    # Choose r ∈ F \ {1,2} (any fixed or random choice is fine for this demo)
    if r is None:
        # simple deterministic pick for reproducibility
        r = 5
    assert r not in (1, 2)

    # Precompute Lagrange basis at r over points {0,1,2}
    lam0, lam1, lam2 = lagrange_basis_at([0, 1, 2], r, p)

    # Evaluate p at 1, 2, r using coefficients cp (linear in cp)
    p1 = poly_eval(cp, 1, p)
    p2 = poly_eval(cp, 2, p)
    pr = poly_eval(cp, r, p)

    # Reconstruct f1(r), f2(r), f3(r) via linear combinations (affine functions)
    # f1(1)=x1, f2(1)=x2, f3(1)=x3
    # f1(2)=p(1), f2(2)=x3, f3(2)=x3 + w + 7     (Fig. 4 wiring)
    f1r = (lam0 * z1 + lam1 * x1 + lam2 * p1) % p
    f2r = (lam0 * z2 + lam1 * x2 + lam2 * x3) % p
    f3r = (lam0 * z3 + lam1 * x3 + lam2 * (x3 + w + 7)) % p

    left = pr
    right = (f1r * f2r + f3r) % p
    check_gate = (left == right)
    check_output_zero = (p2 % p == 0)

    return {
        "r": r, "f1(r)": f1r, "f2(r)": f2r, "f3(r)": f3r,
        "p(r)": pr, "gate_ok": check_gate, "output_ok": check_output_zero,
        "accepted": (check_gate and check_output_zero)
    }

if __name__ == "__main__":
    # Example values over a small field
    P = 97  # |F| > M*degG = 4 is enough here
    x1, x2, x3 = 5, 7, 3
    # Choose w so that v2 = 0: v1 = x1*x2 + x3 = 5*7 + 3 = 38; v2 = v1*x3 + (x3 + w + 7)
    # => v2 = 38*3 + (3 + w + 7) = 114 + 10 + w = 124 + w ≡ 0 (mod 97) ⇒ w = 70
    w = 70

    pi = prover_fig4(x1, x2, x3, w, P, strong_hvzk=True)
    verdict = verifier_fig4(x1, x2, x3, pi, P)
    print("Proof accepted?", verdict["accepted"])
    print("Checks:", {k: v for k, v in verdict.items() if k.endswith("_ok")})
