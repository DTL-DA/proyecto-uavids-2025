"""Prueba de humo del dashboard desplegado (capítulo 9): páginas, predicción en vivo con los 16 modelos y aviso fuera de dominio.

    python verificacion_externa/humo_dashboard.py                                    ->  URL pública de Render
    python verificacion_externa/humo_dashboard.py --url http://localhost:8050       ->  contenedor o servidor local

Usa solo la librería estándar. Llama a los mismos *callbacks* que usa el navegador (`/_dash-update-component`):
  1. las tres páginas responden con código 200;
  2. un flujo de prueba al azar, predicho por cada modelo de `artifacts/`, da probabilidades que suman 1 (± 1e-3);
  3. un flujo absurdo (todos los atributos en 1e9) activa el aviso fuera de dominio.
Guarda el resultado en `experiments/resultados/humo_dashboard.json` (o en `--salida`) y termina con código 1 si algo falla.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

E3 = Path(__file__).resolve().parent.parent
URL = "https://uavids-2025-dashboard.onrender.com"


def pedir(url: str, cuerpo: dict | None = None, espera: int = 120):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    req = urllib.request.Request(url, data=datos, headers={"Content-Type": "application/json"} if datos else {})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=espera) as r:
        contenido = r.read()
        return r.status, contenido, time.perf_counter() - t


def valores_plotly(v) -> list[float]:
    """Plotly 6 serializa los arreglos de NumPy como {"dtype": "f8", "bdata": base64}; las versiones anteriores, como listas."""
    if isinstance(v, dict) and "bdata" in v:
        crudo = base64.b64decode(v["bdata"])
        fmt = {"f8": "d", "f4": "f", "i4": "i", "i8": "q"}[v["dtype"]]
        return list(struct.unpack(f"<{len(crudo) // struct.calcsize(fmt)}{fmt}", crudo))
    return [float(x) for x in v]


def spec(s: str) -> dict:
    i, p = s.rsplit(".", 1)
    return {"id": i, "property": p}


def llamar(url: str, cb: dict, valores: dict) -> tuple[dict, float]:
    """Ejecuta un callback de Dash con los valores dados para sus entradas y estados (por id)."""
    salidas = cb["output"][2:-2].split("...") if cb["output"].startswith("..") else [cb["output"]]
    entradas = [dict(i, value=valores.get(i["id"])) for i in cb["inputs"]]
    cuerpo = {"output": cb["output"], "outputs": [spec(o) for o in salidas] if len(salidas) > 1 else spec(salidas[0]),
              "inputs": entradas, "changedPropIds": [f"{entradas[0]['id']}.{entradas[0]['property']}"],
              "state": [dict(s, value=valores.get(s["id"])) for s in cb.get("state", [])]}
    _, contenido, seg = pedir(url + "/_dash-update-component", cuerpo)
    return json.loads(contenido)["response"], seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--salida", default=str(E3 / "experiments" / "resultados" / "humo_dashboard.json"))
    a = ap.parse_args()
    url = a.url.rstrip("/")

    # El plan gratuito de Render se duerme: la primera petición puede tardar cerca de un minuto en despertarlo.
    _, _, despertar = pedir(url + "/", espera=180)
    paginas = {}
    for p in ["/", "/eda", "/modelos"]:
        codigo, _, seg = pedir(url + p)
        paginas[p] = {"codigo": codigo, "segundos": round(seg, 3)}

    deps = json.loads(pedir(url + "/_dash-dependencies")[1])
    azar = next(c for c in deps if c["inputs"][0]["id"] == "m-azar")
    pred = next(c for c in deps if c["inputs"][0]["id"] == "m-predecir")
    atributos = [i["id"] for i in pred["inputs"] if i["id"].startswith("m-x")]

    r, _ = llamar(url, azar, {"m-azar": 1, "m-clase-azar": "Cualquiera"})
    flujo = {k: v["value"] for k, v in r.items() if k.startswith("m-x")}
    verdad = r.get("m-verdad", {}).get("children", "")

    modelos = sorted(json.loads(p.read_text(encoding="utf-8"))["modelo"] for p in (E3 / "artifacts").glob("*/metadata.json"))
    por_modelo = {}
    for m in modelos:
        r, seg = llamar(url, pred, {"m-predecir": 1, "m-modelo": m, **flujo})
        prob = valores_plotly(r["m-prob"]["figure"]["data"][0]["x"])
        suma = sum(prob) / 100
        por_modelo[m] = {"suma_probabilidades": round(suma, 6), "suma_ok": abs(suma - 1) < 1e-3,
                         "aviso_fuera_de_dominio": r["m-ood"]["is_open"], "texto": r["m-texto"]["children"], "segundos": round(seg, 3)}

    absurdo = {i: 1e9 for i in atributos}
    r, _ = llamar(url, pred, {"m-predecir": 1, "m-modelo": "Original", **absurdo})
    ood = {"aviso_activo": r["m-ood"]["is_open"], "mensaje": r["m-ood"]["children"]}

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=E3, capture_output=True, text=True).stdout.strip()
    ok = all(v["codigo"] == 200 for v in paginas.values()) and all(v["suma_ok"] for v in por_modelo.values()) and ood["aviso_activo"]
    res = {"fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": url, "commit_local": commit,
           "segundos_primera_peticion": round(despertar, 2), "paginas": paginas, "flujo_de_prueba": verdad,
           "modelos": por_modelo, "flujo_absurdo": ood, "todo_ok": ok}
    Path(a.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    lat = sorted(v["segundos"] for v in por_modelo.values())
    print(f"{url} | páginas {[v['codigo'] for v in paginas.values()]} | {sum(v['suma_ok'] for v in por_modelo.values())}/{len(por_modelo)} "
          f"modelos con probabilidades que suman 1 | latencia mediana {lat[len(lat) // 2]:.2f} s | fuera de dominio: {ood['aviso_activo']} | "
          f"{'OK' if ok else 'FALLA'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
