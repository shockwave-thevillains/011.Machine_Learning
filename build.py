"""Menjalankan semua contoh di `examples/`, menyimpan output aslinya, lalu membangun `index.html`.

Pemakaian:
    python build.py                 # jalankan semua contoh + bangun index.html
    python build.py --skip-run      # pakai output yang sudah ada di outputs/
    python build.py --only knn svm  # jalankan ulang sebagian contoh saja
    python build.py --fragment F    # juga tulis versi tanpa <head> (untuk embed)
"""
import argparse
import html
import json
import platform
import re
import subprocess
import sys
import time
from datetime import date
from importlib import metadata
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from algorithms import ALGORITMA, CEK, KATEGORI, PREDIKSI

ROOT = Path(__file__).resolve().parent
EX, OUT = ROOT / "examples", ROOT / "outputs"
MARKER = "\n# === "  # bagian setelah penanda ini (sampel data, visualisasi) tidak ditampilkan
PAKET = ["scikit-learn", "torch", "xgboost", "lightgbm", "catboost", "umap-learn", "mlxtend", "numpy"]


def jalankan(slugs):
    OUT.mkdir(exist_ok=True)
    runs_path = OUT / "_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    for slug in slugs:
        print(f"-> {slug}", flush=True)
        t = time.perf_counter()
        p = subprocess.run([sys.executable, f"{slug}.py"], cwd=EX, capture_output=True, text=True)
        dur = time.perf_counter() - t
        if p.returncode != 0:
            sys.exit(f"Gagal menjalankan {slug}.py:\n{p.stderr}")
        (OUT / f"{slug}.txt").write_text(p.stdout.rstrip() + "\n")
        runs[slug] = round(dur, 1)
    runs_path.write_text(json.dumps(runs, indent=1, sort_keys=True))
    versi = {}
    for pk in PAKET:
        try:
            versi[pk] = metadata.version(pk).split("+")[0]
        except metadata.PackageNotFoundError:
            pass
    env = {"python": platform.python_version(), "paket": versi, "tanggal": date.today().isoformat()}
    (OUT / "_env.json").write_text(json.dumps(env, indent=1))


def esc(s):
    return html.escape(str(s), quote=True)


def kode_tampil(slug):
    src = (EX / f"{slug}.py").read_text()
    punya_plot = "# === VISUALISASI" in src
    return src.split(MARKER)[0].rstrip() + "\n", punya_plot


PRED_MARKER, PRED_CETAK = "# === PREDIKSI DATA BARU ===\n", "--- prediksi data baru ---"


def kode_prediksi(slug):
    src = (EX / f"{slug}.py").read_text()
    bagian = src.split(PRED_MARKER, 1)[1].split(MARKER)[0]
    baris = [b for b in bagian.splitlines() if PRED_CETAK not in b]
    return "\n".join(baris).strip() + "\n"


def pisah_output(teks):
    utama, _, pred = teks.partition(PRED_CETAK)
    return utama.rstrip(), pred.strip("\n")


def ribuan(n):
    return f"{n:,}".replace(",", ".")


def tabel_sampel(slug, akhiran="_sampel", judul="Data yang diolah", kelas="data"):
    p = OUT / f"{slug}{akhiran}.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    head = "<th class=\"rn\">baris</th>" + "".join(f"<th>{esc(k)}</th>" for k in d["kolom"])
    rows = "".join(
        f"<tr><td class=\"rn\">{n}</td>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>"
        for n, r in zip(d["nomor_baris"], d["baris"]))
    meta = f"{ribuan(d['total_baris'])} baris × {d['total_kolom']} kolom · ditampilkan {len(d['baris'])} baris"
    if d["tersembunyi"]:
        a, b, n = d["tersembunyi"]
        meta += f" · {n} kolom ({esc(a)} … {esc(b)}) disembunyikan agar muat"
    ket = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in d["keterangan"].items())
    cat = f'<p class="data-note">{esc(d["catatan"])}</p>' if d["catatan"] else ""
    if akhiran == "_baru_sampel":
        meta = f"{len(d['baris'])} baris ditampilkan" + (f" dari {ribuan(d['total_baris'])}" if d["total_baris"] > len(d["baris"]) else "")
    return (f'<section class="{kelas}"><h3>{judul}</h3><p class="data-meta">{meta}</p>'
            f'<div class="sample-wrap"><table class="sample"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
            f'<dl class="kolom">{ket}</dl>{cat}</section>')


def ambil_kpi(a, teks):
    label, pola = a["kpi"]
    m = re.search(pola, teks)
    if not m:
        sys.exit(f"KPI '{label}' tidak ditemukan di output {a['slug']}")
    nilai = m.group(1)
    return label, nilai


def angka_id(s):
    """0.986 -> 0,986 (format desimal Indonesia) tanpa menyentuh persen / pecahan."""
    return s.replace(".", ",") if re.fullmatch(r"[\d.]+", s) else s


def warna(paradigma):
    if paradigma.startswith("Supervised"):
        return "sup"
    if paradigma.startswith("Unsupervised"):
        return "uns"
    return "oth"


def era(tahun):
    if tahun < 1950:
        return "Sebelum 1950"
    return f"{tahun // 10 * 10}-an"


def render(fragment_only=False):
    env = json.loads((OUT / "_env.json").read_text())
    runs = json.loads((OUT / "_runs.json").read_text())
    fmt = HtmlFormatter(nowrap=True)
    lexer = PythonLexer()

    toc, artikel, baris, eras = {}, [], [], {}
    for a in ALGORITMA:
        slug = a["slug"]
        teks, teks_pred = pisah_output((OUT / f"{slug}.txt").read_text())
        kode, punya_plot = kode_tampil(slug)
        label, nilai = ambil_kpi(a, teks)
        w = warna(a["paradigma"])
        kat = KATEGORI[a["kat"]]
        cari = esc(" ".join([a["nama"], a["alias"], kat, a["tokoh"]]).lower())
        toc.setdefault(a["kat"], []).append(f'<li data-slug="{slug}"><a href="#{slug}">{esc(a["nama"])}</a></li>')
        eras.setdefault(era(a["tahun"]), []).append((a["tahun"], a, w))

        baris.append(
            f'<tr data-slug="{slug}" data-kat="{a["kat"]}" data-q="{cari}">'
            f'<td><a href="#{slug}">{esc(a["nama"])}</a></td><td>{esc(kat)}</td>'
            f'<td class="num">{a["tahun"]}</td><td>{esc(a["data"])}</td>'
            f'<td><span class="kpi-l">{esc(label)}</span> <b class="kpi-v">{esc(angka_id(nilai))}</b></td></tr>'
        )

        sejarah = "".join(f"<p>{esc(p)}</p>" for p in a["sejarah"])
        rumus = "".join(f"<code>{esc(r)}</code>" for r in a.get("rumus", []))
        daftar = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)  # noqa: E731
        fig = ""
        if punya_plot and (OUT / f"{slug}.png").exists():
            fig = (f'<figure><img src="outputs/{slug}.png" alt="Grafik hasil {esc(a["nama"])}" loading="lazy">'
                   f'<figcaption>Grafik dibuat oleh bagian visualisasi di <code>examples/{slug}.py</code>.</figcaption></figure>')
        cek = "".join(f"<li>{esc(x)}</li>" for x in CEK[slug])
        cek_html = f'<section class="cek"><h3>Yang bisa Anda cek dari hasil ini</h3><ul>{cek}</ul></section>'
        artikel.append(f"""
<article class="algo" id="{slug}" data-kat="{a['kat']}" data-q="{cari}">
  <header class="algo-head">
    <div class="algo-title">
      <p class="eyebrow"><span class="kat">{esc(kat)}</span><span class="para p-{w}">{esc(a['paradigma'])}</span></p>
      <h2>{esc(a['nama'])}</h2>
      <p class="alias">{esc(a['alias'])}</p>
    </div>
    <p class="year" aria-label="Tahun">{a['tahun']}</p>
  </header>
  <p class="tokoh"><span>Tokoh</span> {esc(a['tokoh'])}</p>
  <div class="cols">
    <section><h3>Sejarah</h3>{sejarah}</section>
    <section><h3>Fungsi &amp; cara kerja</h3><p>{esc(a['fungsi'])}</p><div class="rumus">{rumus}</div></section>
  </div>
  <div class="facts">
    <section><h4>Dipakai untuk</h4><ul>{daftar(a['kegunaan'])}</ul></section>
    <section><h4>Kelebihan</h4><ul class="plus">{daftar(a['plus'])}</ul></section>
    <section><h4>Kekurangan</h4><ul class="minus">{daftar(a['minus'])}</ul></section>
  </div>
  {tabel_sampel(slug)}
  <section class="run">
    <div class="run-head">
      <h3>Contoh running</h3>
      <span class="file">examples/{slug}.py</span>
      <button class="copy" type="button" data-target="code-{slug}">Salin kode</button>
    </div>
    <pre class="code" id="code-{slug}"><code>{highlight(kode, lexer, fmt)}</code></pre>
    <div class="out">
      <p class="out-head"><span class="out-label">Hasil</span><span class="cmd">$ python examples/{slug}.py</span><span class="dur">{str(runs.get(slug, '–')).replace('.', ',')} dtk</span></p>
      <pre><samp>{esc(teks)}</samp></pre>
    </div>
    {fig}
    {cek_html}
  </section>
  <section class="pred">
    <p class="pred-eyebrow">Langkah berikutnya</p>
    <h3>Prediksi data baru</h3>
    <p class="pred-intro">{esc(PREDIKSI[slug][0])}</p>
    {tabel_sampel(slug, "_baru_sampel", "Data baru", "data data-baru")}
    <div class="run-head">
      <h4>Kode prediksi</h4>
      <span class="file">lanjutan examples/{slug}.py, memakai model yang sudah dilatih di atas</span>
      <button class="copy" type="button" data-target="pred-{slug}">Salin kode</button>
    </div>
    <pre class="code" id="pred-{slug}"><code>{highlight(kode_prediksi(slug), lexer, fmt)}</code></pre>
    <div class="out">
      <p class="out-head"><span class="out-label">Hasil prediksi</span></p>
      <pre><samp>{esc(teks_pred)}</samp></pre>
    </div>
    <p class="baca"><b>Cara membaca:</b> {esc(PREDIKSI[slug][1])}</p>
  </section>
</article>""")

    toc_html = "".join(
        f'<li class="toc-group"><p>{esc(KATEGORI[k])}</p><ul>{"".join(v)}</ul></li>' for k, v in toc.items())
    urut_era = sorted(eras, key=lambda e: (e != "Sebelum 1950", e))
    timeline = "".join(
        f'<div class="era"><p class="era-label">{esc(e)}</p><ul>'
        + "".join(f'<li><a class="chip c-{w}" data-slug="{a["slug"]}" href="#{a["slug"]}"><span class="num">{t}</span> {esc(a["nama"])}</a></li>'
                  for t, a, w in sorted(eras[e], key=lambda x: x[0]))
        + "</ul></div>" for e in urut_era)
    chips = '<button type="button" class="f-chip" data-kat="all" aria-pressed="true">Semua</button>' + "".join(
        f'<button type="button" class="f-chip" data-kat="{k}" aria-pressed="false">{esc(v)}</button>' for k, v in KATEGORI.items())
    paket = " · ".join(f"{k} {v}" for k, v in env["paket"].items())
    n = len(ALGORITMA)
    total = sum(runs.get(a["slug"], 0) for a in ALGORITMA)

    body = TEMPLATE.format(
        css=CSS + fmt_css(), n=n, nkat=len(KATEGORI), toc=toc_html, timeline=timeline, chips=chips,
        rows="".join(baris), articles="".join(artikel), python=env["python"], paket=esc(paket),
        tanggal=env["tanggal"], total=f"{total:.0f}", js=JS)
    if fragment_only:
        return body.replace("<!--/head-->", "")
    head, isi = body.split("<!--/head-->")
    return ('<!doctype html>\n<html lang="id">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</head>\n<body>\n{isi}\n</body>\n</html>\n")


def fmt_css():
    # warna token Pygments: kata kunci biru, string & angka merah, komentar abu-abu
    return """
.code .k,.code .kn,.code .kc,.code .kd,.code .ow,.code .kr{color:var(--blue);font-weight:600}
.code .s,.code .s1,.code .s2,.code .sa,.code .sb,.code .sd,.code .si,.code .se,.code .sc{color:var(--red)}
.code .mi,.code .mf,.code .m,.code .mh,.code .il{color:var(--red)}
.code .c,.code .c1,.code .cm,.code .ch{color:var(--muted);font-style:italic}
.code .nf,.code .nc{color:var(--ink);font-weight:600}
.code .nb,.code .bp{color:var(--blue)}
.code .nd{color:var(--blue)}
"""


CSS = r"""
:root{
  /* Layout: katalog spesimen. Indeks lengket di kiri, lembar tiap algoritma bertumpuk di kanan. */
  --paper:#ffffff; --ink:#000000; --blue:#0000ff; --red:#ff0000;
  --muted:#4b4b63; --line:#d6d6e4; --wash:#f4f4ff; --wash-red:#fff2f2;
  --f-display:"Archivo","Arial Narrow",system-ui,sans-serif;
  --f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --s-1:.8125rem; --s0:1rem; --s1:1.0625rem; --s2:1.25rem; --s3:1.75rem; --s4:2.5rem;
  color-scheme:light;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:7.5rem}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:var(--s0);line-height:1.6;-webkit-text-size-adjust:100%}
a{color:var(--blue);text-underline-offset:.18em}
a:hover{color:var(--red)}
:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
code,pre,samp,.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
h1,h2,h3,h4{text-wrap:balance;margin:0}
img{max-width:100%;height:auto}

.wrap{max-width:1320px;margin:0 auto;padding-inline:16px}
@media (min-width:720px){.wrap{padding-inline:32px}}

/* ---------- kepala halaman ---------- */
.masthead{padding-block:3rem 2rem;border-bottom:3px solid var(--ink)}
.kicker{font-family:var(--f-mono);font-size:var(--s-1);letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin:0 0 1rem}
.masthead h1{font-family:var(--f-display);font-stretch:72%;font-weight:850;font-size:clamp(2.6rem,7.4vw,5.6rem);line-height:.92;letter-spacing:-.01em;max-width:14ch}
.masthead h1 em{font-style:normal;color:var(--blue)}
.masthead h1 .dot{color:var(--red)}
.lede{font-size:var(--s2);line-height:1.5;max-width:62ch;margin:1.5rem 0 0}
.stats{display:flex;flex-wrap:wrap;gap:1rem 2.5rem;margin:2rem 0 0;padding:0;list-style:none}
.stats li{display:flex;flex-direction:column}
.stats b{font-family:var(--f-display);font-stretch:75%;font-weight:800;font-size:var(--s4);line-height:1;color:var(--red)}
.stats span{font-size:var(--s-1);color:var(--muted)}
.env{font-family:var(--f-mono);font-size:var(--s-1);color:var(--muted);margin:1.5rem 0 0;max-width:90ch}

/* ---------- bar filter ---------- */
.filterbar{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--paper);border-bottom:1px solid var(--ink)}
.filterbar .wrap{display:flex;flex-wrap:wrap;gap:.6rem 1rem;align-items:center;padding-block:.7rem}
.search{flex:1 1 220px;max-width:340px;display:flex;align-items:center;gap:.5rem;border:1px solid var(--ink);padding:.35rem .6rem}
.search:focus-within{outline:2px solid var(--blue);outline-offset:1px}
.search input{border:0;outline:0;font:inherit;width:100%;background:transparent;color:var(--ink)}
.search svg{flex:none}
.f-chips{display:flex;gap:.4rem;overflow-x:auto;flex:1 1 480px;min-width:0;padding-bottom:2px;scrollbar-width:thin}
.f-chip{flex:none;font:500 var(--s-1)/1 var(--f-body);border:1px solid var(--ink);background:var(--paper);color:var(--ink);padding:.45rem .65rem;cursor:pointer;border-radius:2px}
.f-chip:hover{border-color:var(--blue);color:var(--blue)}
.f-chip[aria-pressed="true"]{background:var(--blue);border-color:var(--blue);color:#fff}
.count{font-family:var(--f-mono);font-size:var(--s-1);color:var(--muted);white-space:nowrap}

/* ---------- tata letak utama ---------- */
.layout{display:grid;grid-template-columns:minmax(0,1fr);gap:3rem}
@media (min-width:1180px){.layout{grid-template-columns:230px minmax(0,1fr)}}
.toc{display:none}
@media (min-width:1180px){
  .toc{display:block;position:sticky;top:5.2rem;align-self:start;max-height:calc(100vh - 6rem);overflow:auto;padding-block:2rem;font-size:var(--s-1)}
}
.toc ul{list-style:none;margin:0;padding:0}
.toc-group{margin-bottom:1.1rem}
.toc-group>p{font-family:var(--f-mono);text-transform:uppercase;letter-spacing:.08em;font-size:.72rem;color:var(--red);margin:0 0 .3rem}
.toc a{color:var(--ink);text-decoration:none;display:block;padding:.12rem 0}
.toc a:hover{color:var(--blue)}
.main{min-width:0;padding-block:2.5rem 4rem}

.block-title{font-family:var(--f-display);font-stretch:78%;font-weight:800;font-size:var(--s3);margin:0 0 .4rem}
.block-sub{color:var(--muted);margin:0 0 1.4rem;max-width:70ch}
.block{margin-bottom:4rem}

/* ---------- lini masa ---------- */
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;font-size:var(--s-1);margin:0 0 1.2rem;padding:0;list-style:none}
.legend li::before,.chip::before{content:"";display:inline-block;width:.6em;height:.6em;margin-right:.45em;vertical-align:.05em;background:var(--c)}
.c-sup,.l-sup{--c:var(--blue)} .c-uns,.l-uns{--c:var(--red)} .c-oth,.l-oth{--c:var(--ink)}
.era{display:grid;grid-template-columns:minmax(0,1fr);gap:.4rem;padding-block:.8rem;border-top:1px solid var(--line)}
@media (min-width:720px){.era{grid-template-columns:9.5rem minmax(0,1fr);gap:1rem}}
.era-label{font-family:var(--f-mono);font-size:var(--s-1);color:var(--muted);margin:.25rem 0 0}
.era ul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:.4rem}
.chip{display:inline-block;font-size:var(--s-1);color:var(--ink);text-decoration:none;border:1px solid var(--line);padding:.25rem .55rem;border-radius:2px;transition:opacity .2s}
.chip .num{color:var(--muted)}
.chip:hover{border-color:var(--blue);color:var(--blue)}
.chip.dim{opacity:.25}

/* ---------- tabel ringkasan ---------- */
.tablewrap{overflow-x:auto;border-top:2px solid var(--ink)}
table{border-collapse:collapse;width:100%;min-width:760px;font-size:.9rem}
th{font-family:var(--f-mono);font-weight:500;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;text-align:left;color:var(--muted);padding:.6rem .7rem;border-bottom:1px solid var(--ink)}
td{padding:.5rem .7rem;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:hover{background:var(--wash)}
td a{color:var(--ink);font-weight:600;text-decoration:none}
td a:hover{color:var(--blue);text-decoration:underline}
.kpi-l{color:var(--muted)}
.kpi-v{font-family:var(--f-mono);color:var(--red);white-space:nowrap}

/* ---------- lembar algoritma ---------- */
.algo{border-top:3px solid var(--ink);padding-block:2rem 3.5rem}
.algo-head{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
.algo-title{min-width:0}
.eyebrow{display:flex;flex-wrap:wrap;gap:.3rem .9rem;font-family:var(--f-mono);font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;margin:0 0 .5rem}
.kat{color:var(--blue)}
.para{color:var(--muted)}
.algo h2{font-family:var(--f-display);font-stretch:78%;font-weight:800;font-size:clamp(1.9rem,4.2vw,2.9rem);line-height:1}
.alias{color:var(--muted);margin:.35rem 0 0;font-size:var(--s1)}
.year{font-family:var(--f-display);font-stretch:70%;font-weight:300;font-size:clamp(2.4rem,6vw,4.2rem);line-height:.9;color:var(--red);margin:0;font-variant-numeric:tabular-nums}
.tokoh{font-size:.92rem;margin:1rem 0 0;max-width:90ch}
.tokoh span{font-family:var(--f-mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--red);margin-right:.4rem}
.cols{display:grid;grid-template-columns:minmax(0,1fr);gap:1.5rem 3rem;margin-top:1.8rem}
@media (min-width:900px){.cols{grid-template-columns:repeat(2,minmax(0,1fr))}}
.cols h3,.run h3,.data h3,.cek h3{font-family:var(--f-display);font-stretch:85%;font-weight:700;font-size:var(--s2);margin:0 0 .5rem}
.cols p{margin:0 0 .8rem;max-width:65ch}
.rumus{display:flex;flex-direction:column;gap:.4rem;margin-top:.9rem}
.rumus code{display:block;background:var(--wash);padding:.55rem .75rem;font-size:.86rem;overflow-x:auto;white-space:pre-wrap}
.facts{display:grid;grid-template-columns:minmax(0,1fr);gap:1.2rem 2rem;margin-top:1.6rem;padding-top:1.2rem;border-top:1px solid var(--line)}
@media (min-width:720px){.facts{grid-template-columns:repeat(3,minmax(0,1fr))}}
.facts h4{font-family:var(--f-mono);font-weight:500;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .4rem}
.facts ul{margin:0;padding:0;list-style:none;font-size:.93rem}
.facts li{padding-left:1.1rem;position:relative;margin-bottom:.25rem}
.facts li::before{content:"–";position:absolute;left:0;color:var(--muted)}
.facts .plus li::before{content:"+";color:var(--blue);font-weight:700}
.facts .minus li::before{content:"−";color:var(--red);font-weight:700}

.run{margin-top:2rem}
.run-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem 1rem;margin-bottom:.6rem}
.run-head h3{margin:0}
.file{font-family:var(--f-mono);font-size:var(--s-1);color:var(--muted);overflow-wrap:anywhere}
.copy{margin-left:auto;font:500 var(--s-1)/1 var(--f-body);background:var(--paper);color:var(--blue);border:1px solid var(--blue);padding:.4rem .65rem;cursor:pointer;border-radius:2px}
.copy:hover{background:var(--blue);color:#fff}
pre{margin:0;overflow-x:auto;font-size:.82rem;line-height:1.55}
.code{background:var(--wash);border:1px solid var(--line);padding:1rem 1.1rem;max-height:30rem;overflow:auto}
.out{margin-top:1rem;border:1px solid var(--ink)}
.out-head{display:flex;flex-wrap:wrap;gap:.3rem 1rem;align-items:baseline;margin:0;padding:.45rem .9rem;border-bottom:1px solid var(--ink);font-size:var(--s-1)}
.out-label{font-family:var(--f-mono);text-transform:uppercase;letter-spacing:.08em;color:var(--red);font-weight:600}
.cmd{font-family:var(--f-mono);color:var(--ink);overflow-wrap:anywhere}
.dur{font-family:var(--f-mono);color:var(--muted);margin-left:auto}
.out pre{padding:.9rem 1.1rem}
figure{margin:1rem 0 0;max-width:760px}
figure img{display:block;border:1px solid var(--line)}
figcaption{font-size:var(--s-1);color:var(--muted);margin-top:.35rem}
.data{margin-top:2rem}
.data-meta{font-family:var(--f-mono);font-size:var(--s-1);color:var(--muted);margin:0 0 .6rem}
.sample-wrap{overflow-x:auto;border-top:2px solid var(--ink)}
table.sample{min-width:0;width:auto;font-family:var(--f-mono);font-size:.8rem;font-variant-numeric:tabular-nums}
.sample th{text-transform:none;letter-spacing:0;font-size:.78rem;color:var(--blue);font-weight:600;white-space:nowrap;padding:.45rem .7rem}
.sample td{white-space:nowrap;padding:.35rem .7rem}
.sample .rn{color:var(--muted);font-weight:400}
.kolom{display:grid;grid-template-columns:minmax(0,1fr);gap:.15rem 1.2rem;margin:1rem 0 0;font-size:.9rem;max-width:90ch}
@media (min-width:720px){.kolom{grid-template-columns:minmax(9rem,max-content) minmax(0,1fr)}}
.kolom dt{font-family:var(--f-mono);font-size:.8rem;color:var(--blue);padding-top:.15rem}
.kolom dd{margin:0 0 .45rem}
.data-note{font-size:.88rem;color:var(--muted);margin:.4rem 0 0;max-width:80ch}
.cek{margin-top:1.4rem;padding:1rem 1.2rem;background:var(--wash-red);max-width:90ch}
.cek h3{color:var(--red)}
.cek ul{list-style:none;margin:0;padding:0;font-size:.94rem}
.cek li{position:relative;padding-left:1.4rem;margin-bottom:.45rem}
.cek li::before{content:"→";position:absolute;left:0;color:var(--red);font-weight:700}
.pred{margin-top:2.4rem;padding-top:1.4rem;border-top:2px dashed var(--blue)}
.pred-eyebrow{font-family:var(--f-mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--blue);margin:0 0 .25rem}
.pred h3{font-family:var(--f-display);font-stretch:85%;font-weight:700;font-size:var(--s2);margin:0 0 .5rem;color:var(--blue)}
.pred h4{font-family:var(--f-display);font-stretch:85%;font-weight:700;font-size:var(--s1);margin:0}
.pred-intro{margin:0;max-width:75ch}
.pred .data{margin-top:1.2rem}
.pred .data h3{color:var(--ink);font-size:var(--s1)}
.pred .run-head{margin-top:1.4rem}
.baca{margin:1rem 0 0;padding:.8rem 1rem;background:var(--wash);font-size:.94rem;max-width:90ch}
.baca b{color:var(--blue)}

.howto ol{padding-left:1.2rem;max-width:72ch}
.howto li{margin-bottom:.6rem}
.howto pre{background:var(--wash);border:1px solid var(--line);padding:.7rem .9rem;margin-top:.4rem}
.foot{border-top:3px solid var(--ink);padding-block:1.5rem 3rem;font-size:var(--s-1);color:var(--muted)}
.empty{padding:2rem 0;color:var(--muted)}
"""

TEMPLATE = """<title>Atlas Algoritma Machine Learning</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..900&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap">
<style>{css}</style>
<!--/head-->
<header class="masthead">
  <div class="wrap">
    <p class="kicker">Katalog · sejarah · fungsi · data · kode · hasil asli</p>
    <h1>Atlas Algoritma <em>Machine Learning</em><span class="dot">.</span></h1>
    <p class="lede">{n} algoritma machine learning populer, dari kuadrat terkecil Legendre (1805) sampai Transformer (2017). Setiap lembar berisi sejarah singkat, cara kerja, kegunaan, contoh data yang diolah beserta arti kolomnya, kode Python, output yang benar-benar keluar saat kode itu dijalankan, panduan apa yang bisa Anda cek dari hasilnya, lalu contoh memakai model itu untuk memprediksi data baru.</p>
    <ul class="stats">
      <li><b class="num">{n}</b><span>algoritma</span></li>
      <li><b class="num">{nkat}</b><span>kategori</span></li>
      <li><b class="num">{n}</b><span>skrip dijalankan</span></li>
      <li><b class="num">{total}</b><span>detik total eksekusi</span></li>
    </ul>
    <p class="env">Dijalankan {tanggal} · Python {python} · {paket} · CPU 4 core, tanpa GPU</p>
  </div>
</header>

<nav class="filterbar" aria-label="Filter algoritma">
  <div class="wrap">
    <label class="search" for="q">
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M11 11l4 4" stroke="currentColor" stroke-width="1.6"/></svg>
      <input id="q" type="search" placeholder="Cari algoritma atau tokoh…" autocomplete="off">
    </label>
    <div class="f-chips" role="group" aria-label="Kategori">{chips}</div>
    <span class="count" id="count">{n} dari {n}</span>
  </div>
</nav>

<div class="wrap layout">
  <aside class="toc" aria-label="Daftar isi"><ul>{toc}</ul></aside>
  <main class="main">
    <section class="block" id="lini-masa">
      <h2 class="block-title">Lini masa</h2>
      <p class="block-sub">Tahun kelahiran tiap algoritma, dikelompokkan per dekade. Klik untuk melompat ke lembarnya.</p>
      <ul class="legend"><li class="l-sup">Supervised</li><li class="l-uns">Unsupervised</li><li class="l-oth">Reinforcement &amp; self-supervised</li></ul>
      {timeline}
    </section>

    <section class="block" id="ringkasan">
      <h2 class="block-title">Ringkasan hasil</h2>
      <p class="block-sub">Satu angka kunci dari output tiap contoh. Dataset dan metriknya berbeda-beda, jadi angka antar-baris tidak untuk dibandingkan langsung.</p>
      <div class="tablewrap"><table>
        <thead><tr><th>Algoritma</th><th>Kategori</th><th>Tahun</th><th>Data contoh</th><th>Hasil utama</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </section>

    <div id="daftar">{articles}</div>
    <p class="empty" id="empty" hidden>Tidak ada algoritma yang cocok. Coba kata kunci lain atau pilih kategori "Semua".</p>

    <section class="block howto" id="jalankan">
      <h2 class="block-title">Menjalankan sendiri</h2>
      <ol>
        <li>Pasang dependensi:<pre><code>pip install -r requirements.txt</code></pre></li>
        <li>Jalankan contoh mana pun dari folder <code>examples/</code>:<pre><code>cd examples
python kmeans.py</code></pre></li>
        <li>Bangun ulang halaman ini (menjalankan semua contoh, menyimpan output ke <code>outputs/</code>, menulis <code>index.html</code>):<pre><code>python build.py</code></pre></li>
      </ol>
      <p>Angka bisa sedikit berbeda di mesin lain karena versi library, jumlah thread, dan perangkat keras. Waktu eksekusi sangat bergantung pada mesin.</p>
    </section>
  </main>
</div>
<footer class="foot"><div class="wrap">Atlas Algoritma Machine Learning · dibangun otomatis oleh <code>build.py</code> dari {n} skrip di <code>examples/</code>.</div></footer>
<script>{js}</script>"""

JS = r"""
(function(){
  var state={kat:"all",q:""};
  var arts=[].slice.call(document.querySelectorAll(".algo"));
  var rows=[].slice.call(document.querySelectorAll("#ringkasan tbody tr"));
  var tocs=[].slice.call(document.querySelectorAll(".toc li[data-slug]"));
  var chips=[].slice.call(document.querySelectorAll(".chip"));
  var groups=[].slice.call(document.querySelectorAll(".toc-group"));
  var count=document.getElementById("count"), empty=document.getElementById("empty");
  function cocok(el){
    return (state.kat==="all"||el.getAttribute("data-kat")===state.kat) &&
           (!state.q||el.getAttribute("data-q").indexOf(state.q)>-1);
  }
  function apply(){
    var tampil={}, n=0;
    arts.forEach(function(a){var ok=cocok(a); a.hidden=!ok; if(ok){tampil[a.id]=1;n++;}});
    rows.forEach(function(r){r.hidden=!tampil[r.getAttribute("data-slug")];});
    tocs.forEach(function(t){t.hidden=!tampil[t.getAttribute("data-slug")];});
    groups.forEach(function(g){g.hidden=!g.querySelector("li[data-slug]:not([hidden])");});
    chips.forEach(function(c){c.classList.toggle("dim",!tampil[c.getAttribute("data-slug")]);});
    count.textContent=n+" dari "+arts.length;
    empty.hidden=n>0;
  }
  document.querySelectorAll(".f-chip").forEach(function(b){
    b.addEventListener("click",function(){
      state.kat=b.getAttribute("data-kat");
      document.querySelectorAll(".f-chip").forEach(function(x){x.setAttribute("aria-pressed",x===b?"true":"false");});
      apply();
    });
  });
  document.getElementById("q").addEventListener("input",function(e){state.q=e.target.value.trim().toLowerCase();apply();});
  document.querySelectorAll(".copy").forEach(function(btn){
    btn.addEventListener("click",function(){
      var pre=document.getElementById(btn.getAttribute("data-target"));
      var text=pre.innerText;
      function done(msg){btn.textContent=msg;setTimeout(function(){btn.textContent="Salin kode";},1600);}
      function pilih(){var r=document.createRange();r.selectNodeContents(pre);var s=getSelection();s.removeAllRanges();s.addRange(r);done("Teks terpilih, tekan Ctrl+C");}
      try{navigator.clipboard.writeText(text).then(function(){done("Tersalin");},pilih);}catch(e){pilih();}
    });
  });
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fragment")
    args = ap.parse_args()
    slugs = [a["slug"] for a in ALGORITMA]
    if args.only:
        jalankan(args.only)
    elif not args.skip_run:
        jalankan(slugs)
    (ROOT / "index.html").write_text(render())
    if args.fragment:
        Path(args.fragment).write_text(render(fragment_only=True))
    print(f"index.html ditulis ({len(slugs)} algoritma)")


if __name__ == "__main__":
    main()
