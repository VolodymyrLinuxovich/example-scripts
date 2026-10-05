"""Memory-light equivalent of agents.code.data.build_full_datasets (pyarrow streaming).

Writes full.parquet, downsampled_full.parquet (every 4th era) and the matching benchmark files
into numerai/v5.3/ after train/validation/benchmark parquets have been downloaded.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

D = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "v5.3") + "/"
STEP = 4

tr, va = pq.ParquetFile(D + "train.parquet"), pq.ParquetFile(D + "validation.parquet")
cols = [c for c in tr.schema_arrow.names if c != "data_type"]
schema = pa.schema([tr.schema_arrow.field(c) for c in cols]).remove_metadata()

eras = set(pq.read_table(D + "train.parquet", columns=["era"])["era"].to_pylist())
v = pq.read_table(D + "validation.parquet", columns=["era", "data_type"])
v = v.filter(pc.equal(v["data_type"], "validation"))
eras |= set(v["era"].to_pylist())
eras = sorted(eras, key=int)
keep = pa.array([e for i, e in enumerate(eras) if i % STEP == 0])
print(f"{len(eras)} eras total ({eras[0]}..{eras[-1]}), keeping {len(keep)} downsampled", flush=True)

full_w = pq.ParquetWriter(D + "full.parquet", schema, compression="zstd")
down_w = pq.ParquetWriter(D + "downsampled_full.parquet", schema, compression="zstd")
n_full = n_down = 0
for src, is_val in ((tr, False), (va, True)):
    read_cols = cols + (["data_type"] if is_val else [])
    for batch in src.iter_batches(batch_size=50_000, columns=read_cols):
        t = pa.Table.from_batches([batch])
        if is_val:
            t = t.filter(pc.equal(t["data_type"], "validation"))
        t = t.select(cols).cast(schema)
        if t.num_rows == 0:
            continue
        full_w.write_table(t)
        n_full += t.num_rows
        d = t.filter(pc.is_in(t["era"], keep))
        if d.num_rows:
            down_w.write_table(d)
            n_down += d.num_rows
    print("done source", "validation" if is_val else "train", n_full, n_down, flush=True)
full_w.close()
down_w.close()

# benchmark models (small; pandas is fine)
val = pd.read_parquet(D + "validation.parquet", columns=["data_type"])  # id is the index
val_ids = set(val.index[val["data_type"] == "validation"])
btr = pd.read_parquet(D + "train_benchmark_models.parquet")
bva = pd.read_parquet(D + "validation_benchmark_models.parquet")
for b in (btr, bva):
    if "id" in b.columns:
        b.set_index("id", inplace=True)
bva = bva.loc[bva.index.isin(val_ids)]
bfull = pd.concat([btr, bva])
bfull.to_parquet(D + "full_benchmark_models.parquet")
down_ids = set(pq.read_table(D + "downsampled_full.parquet", columns=["id"])["id"].to_pylist())
bfull.loc[bfull.index.isin(down_ids)].to_parquet(D + "downsampled_full_benchmark_models.parquet")
print("rows full", n_full, "down", n_down, "bench", len(bfull), flush=True)
