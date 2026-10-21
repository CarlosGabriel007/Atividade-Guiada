import random
from collections import deque
from heapq import heappush, heappop
from time import perf_counter

OBJETIVO = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # 0 = espaco vazio
POS_OBJ = {v: i for i, v in enumerate(OBJETIVO)}

def vizinhos(estado):
    """Gera (estado_filho, custo=1) 
    i = estado.index(0)
    r, c = divmod(i, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            lst = list(estado)
            lst[i], lst[j] = lst[j], lst[i]
            yield tuple(lst), 1

def h_zero(e):
    return 0

def h_fora_do_lugar(e):
    return sum(1 for i, v in enumerate(e) if v != 0 and v != OBJETIVO[i])

def h_manhattan(e):
    t = 0
    for i, v in enumerate(e):
        if v != 0:
            j = POS_OBJ[v]
            t += abs(i // 3 - j // 3) + abs(i % 3 - j % 3)
    return t

def h_dobro_manhattan(e):
    """Heuristica INADMISSIVEL de proposito: 2 x Manhattan."""
    return 2 * h_manhattan(e)

def reconstroi(pai, no):
    cam = [no]
    while pai[no] is not None:
        no = pai[no]
        cam.append(no)
    return cam[::-1]

def a_estrela(inicio, h):
    """A* em grafo com atualizacao de g/predecessor (exclusao preguicosa).
    Retorna (custo, caminho, expansoes, reaberturas)."""
    cont = 0
    fila = [(h(inicio), cont, 0, inicio)]
    melhor_g = {inicio: 0}
    pai = {inicio: None}
    expandidos = set()
    expansoes = reaberturas = 0
    while fila:
        _, _, g, atual = heappop(fila)
        if g != melhor_g[atual]:
            continue
        if atual == OBJETIVO:
            return g, reconstroi(pai, atual), expansoes, reaberturas
        if atual in expandidos:
            reaberturas += 1
        expandidos.add(atual)
        expansoes += 1
        for viz, c in vizinhos(atual):
            ng = g + c
            if ng < melhor_g.get(viz, float("inf")):
                melhor_g[viz] = ng
                pai[viz] = atual
                cont += 1
                heappush(fila, (ng + h(viz), cont, ng, viz))
    return None, None, expansoes, reaberturas

def gulosa(inicio, h):
    """Busca gulosa (f = h) em grafo com lista de explorados."""
    cont = 0
    fila = [(h(inicio), cont, 0, inicio)]
    pai = {inicio: None}
    explorado = set()
    expansoes = 0
    while fila:
        _, _, g, atual = heappop(fila)
        if atual in explorado:
            continue
        if atual == OBJETIVO:
            return g, reconstroi(pai, atual), expansoes, 0
        explorado.add(atual)
        expansoes += 1
        for viz, c in vizinhos(atual):
            if viz not in explorado and viz not in pai:
                pai[viz] = atual
                cont += 1
                heappush(fila, (h(viz), cont, g + c, viz))
    return None, None, expansoes, 0

def custo_otimo_todos():
    """h*(n) exato para os 181.440 estados alcancaveis (BFS do objetivo;
    os movimentos sao reversiveis e de custo 1)."""
    dist = {OBJETIVO: 0}
    fila = deque([OBJETIVO])
    while fila:
        u = fila.popleft()
        for v, _ in vizinhos(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                fila.append(v)
    return dist

def tempo_ms(func, *args, repeticoes=1):
    t0 = perf_counter()
    for _ in range(repeticoes):
        func(*args)
    return (perf_counter() - t0) / repeticoes * 1000

def mostra(e):
    return " / ".join("".join(str(x) if x else "_" for x in e[i:i + 3])
                      for i in (0, 3, 6))

def main():
    hstar = custo_otimo_todos()
    print("Estados alcancaveis:", len(hstar), "| pior caso h* =", max(hstar.values()))

    heur = [("h = 0", h_zero), ("pecas fora do lugar", h_fora_do_lugar),
            ("Manhattan", h_manhattan), ("2 x Manhattan", h_dobro_manhattan)]

    print("\n[Teste] admissibilidade e consistencia em TODOS os estados")
    for nome, h in heur:
        viol_adm = sum(1 for e, d in hstar.items() if h(e) > d)
        viol_con = 0
        for e in hstar:
            he = h(e)
            for v, c in vizinhos(e):
                if he > c + h(v):
                    viol_con += 1
        print("  {:<22} violacoes de admissibilidade: {:>6} | de consistencia: {}".format(
            nome, viol_adm, viol_con))

    rng = random.Random(2026)
    por_prof = {}
    for e, d in sorted(hstar.items()):
        por_prof.setdefault(d, []).append(e)

    # Instancias fixas (seed 2026): uma por profundidade otima
    instancias = {d: rng.choice(por_prof[d]) for d in (8, 14, 20)}

    print("\n[Etapa 1] f dos 3 filhos da raiz (espaco vazio na borda), custo otimo 7:")
    rng_e1 = random.Random(7)  # gerador separado: nao altera as outras instancias
    candidatas = [e for e in por_prof[7] if e.index(0) in (1, 3, 5, 7)]
    raiz = rng_e1.choice(candidatas)
    print("  raiz:", mostra(raiz), "| h1 =", h_fora_do_lugar(raiz),
          "| h2 =", h_manhattan(raiz), "| h*(raiz) =", hstar[raiz])
    for viz, c in vizinhos(raiz):
        print("   filho {}  g={}  h1={}  h2={}  f1={}  f2={}  h*={}".format(
            mostra(viz), c, h_fora_do_lugar(viz), h_manhattan(viz),
            c + h_fora_do_lugar(viz), c + h_manhattan(viz), hstar[viz]))

    print("\n[Resultados por instancia]")
    cab = "{:<22}{:>7}{:>10}{:>7}{:>10}"
    for d, ini in instancias.items():
        print("\n  Instancia custo otimo {} : {}".format(d, mostra(ini)))
        print("  " + cab.format("Algoritmo", "Custo", "Expans.", "Reab.", "Tempo(ms)"))
        for nome, tipo, h in (("Gulosa (Manhattan)", "g", h_manhattan),
                              ("A* h=0", "a", h_zero),
                              ("A* fora do lugar", "a", h_fora_do_lugar),
                              ("A* Manhattan", "a", h_manhattan),
                              ("A* 2xManhattan", "a", h_dobro_manhattan)):
            f = gulosa if tipo == "g" else a_estrela
            custo, cam, exp, reab = f(ini, h)
            t = tempo_ms(f, ini, h)
            marca = "" if custo == d else "  <- NAO OTIMO"
            print("  " + cab.format(nome, custo, exp, reab, "{:.1f}".format(t)) + marca)

    print("\n[Media em 30 instancias com custo otimo 20 (seed 2026)]")
    amostra = rng.sample(por_prof[20], 30)
    print("  {:<22}{:>8}{:>12}{:>8}{:>12}".format(
        "Algoritmo", "Nao-otim", "Expans.med", "Reab.", "Tempo med(ms)"))
    for nome, tipo, h in (("Gulosa (Manhattan)", "g", h_manhattan),
                          ("A* h=0", "a", h_zero),
                          ("A* fora do lugar", "a", h_fora_do_lugar),
                          ("A* Manhattan", "a", h_manhattan),
                          ("A* 2xManhattan", "a", h_dobro_manhattan)):
        f = gulosa if tipo == "g" else a_estrela
        nao_otimos = exps = reabs = 0
        t0 = perf_counter()
        for ini in amostra:
            custo, _, exp, reab = f(ini, h)
            nao_otimos += custo != 20
            exps += exp
            reabs += reab
        t = (perf_counter() - t0) / len(amostra) * 1000
        print("  {:<22}{:>8}{:>12.0f}{:>8}{:>12.1f}".format(
            nome, nao_otimos, exps / len(amostra), reabs, t))

    print("\n[Etapa 3] contraexemplo da gulosa (primeiro achado, seed 2026):")
    exemplos = [e for e in rng.sample(por_prof[14], 200)
                if gulosa(e, h_manhattan)[0] > 14]
    for e in exemplos[:1]:
        custo, cam, exp, _ = gulosa(e, h_manhattan)
        ca, _, ea, _ = a_estrela(e, h_manhattan)
        print("  inicio:", mostra(e), "| otimo =", hstar[e])
        print("  gulosa: custo", custo, "em", exp, "expansoes")
        print("  A*    : custo", ca, "em", ea, "expansoes")
    nao = len(exemplos)
    print("  entre 200 instancias de custo 14, a gulosa nao foi otima em", nao)

    print("\n[Etapa 2/limite] pecas fora do lugar vs Manhattan no estado de custo 20:")
    ini = instancias[20]
    print("  h1 =", h_fora_do_lugar(ini), "| h2 =", h_manhattan(ini),
          "| h* =", hstar[ini])

if __name__ == "__main__":
    main()
