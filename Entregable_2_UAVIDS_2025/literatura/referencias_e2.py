"""Metadatos de las referencias nuevas del Entregable 2, consultados en OpenAlex por DOI y guardados en referencias_e2.json.
Las referencias sin DOI se registran a mano en el cuaderno, con su fuente verificada."""
import json
import time
import urllib.request
from pathlib import Path

CARPETA = Path(__file__).parent
DOIS = [
    "10.1109/cvpr.2015.7298878", "10.1207/s15516709cog1402_1", "10.3115/v1/d14-1179", "10.1126/science.1127647", "10.1016/j.neunet.2005.06.042", "10.1162/neco.2006.18.7.1527",
    "10.48550/arxiv.1706.03762", "10.1109/tnn.2008.2005605", "10.1109/cvpr.2016.90", "10.1145/3422622", "10.1609/aaai.v35i8.16826", "10.48550/arxiv.2404.19756",
    "10.48550/arxiv.2106.11959", "10.1109/72.279181", "10.1613/jair.1.12125", "10.1080/00401706.1999.10485670", "10.48550/arxiv.1703.01365", "10.48550/arxiv.1612.01474",
    "10.48550/arxiv.1506.02142", "10.1561/2200000101", "10.1109/tit.1967.1053964", "10.3982/ecta5771", "10.1162/089976699300016007", "10.1080/03610928008827904",
    "10.1214/aoms/1177731915", "10.1007/bf02289823", "10.1080/01621459.1974.10482955", "10.1002/sim.4780050506", "10.48550/arxiv.1706.04599", "10.48550/arxiv.2004.11362",
    "10.48550/arxiv.2010.01028", "10.1109/cvpr42600.2020.00975", "10.48550/arxiv.2202.08906", "10.1007/s11263-023-01913-8", "10.48550/arxiv.2006.10726",
    "10.1016/j.media.2020.101907", "10.1016/j.neucom.2025.129414", "10.1016/j.patcog.2025.112325", "10.1002/cpe.71022", "10.1002/9781118548387",
    "10.1177/0272989x06295361",
]


CORRECCIONES = {
    "10.1080/03610928008827904": {
        "titulo": "Approximations of the critical region of the Friedman statistic",
        "fuente": "Communications in Statistics - Theory and Methods"
    },
    "10.1080/01621459.1974.10482955": {
        "paginas": "364-367"
    },
    "10.1080/00401706.1999.10485670": {
        "paginas": "212-223"
    },
    "10.48550/arxiv.2006.10726": {
        "autores": [
            "Dequan Wang",
            "Evan Shelhamer",
            "Shaoteng Liu",
            "Bruno A. Olshausen",
            "Trevor Darrell"
        ]
    },
    "10.1109/cvpr.2015.7298878": {
        "fuente": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"
    },
    "10.1109/cvpr.2016.90": {
        "fuente": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"
    },
    "10.1109/cvpr42600.2020.00975": {
        "fuente": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
    }
}


def main():
    ruta = CARPETA / "referencias_e2.json"
    refs = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for doi in DOIS:
        if doi in refs:
            continue
        url = f"https://api.openalex.org/works/doi:{doi}?select=doi,title,publication_year,authorships,primary_location,biblio,type"
        try:
            w = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
        except Exception as e:
            print("no encontrado", doi, e)
            continue
        fuente = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        refs[doi] = {"titulo": w["title"], "anio": w["publication_year"], "autores": [a["author"]["display_name"] for a in w.get("authorships", [])], "fuente": fuente,
                     "volumen": (w.get("biblio") or {}).get("volume"), "numero": (w.get("biblio") or {}).get("issue"),
                     "paginas": "-".join(x for x in [(w.get("biblio") or {}).get("first_page"), (w.get("biblio") or {}).get("last_page")] if x), "tipo": w.get("type")}
        print(doi, "|", refs[doi]["anio"], "|", "; ".join(refs[doi]["autores"][:3]), "|", (refs[doi]["titulo"] or "")[:70], "|", fuente, flush=True)
        time.sleep(1)
    # Correcciones verificadas de metadatos que OpenAlex trae mal (título corrupto, páginas, autor invertido, fuente duplicada)
    for doi, cambios in CORRECCIONES.items():
        if doi in refs:
            refs[doi].update(cambios)
    for m in refs.values():
        if m.get("fuente") == "arXiv (Cornell University)":
            m["fuente"] = "arXiv"
    ruta.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
