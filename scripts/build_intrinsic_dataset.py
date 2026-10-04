import os
import glob
import pandas as pd # type: ignore
from rdkit import Chem # type: ignore

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/intrinsic_pka_master.csv"

def parse_sdf(path):
    records = []
    suppl = Chem.SDMolSupplier(path, removeHs=False)

    for mol in suppl:
        if mol is None:
            continue

        if not mol.HasProp("pKa"):
            continue

        try:
            pka = float(mol.GetProp("pKa"))
        except:
            continue

        smiles = Chem.MolToSmiles(mol, canonical=True)

        records.append({
            "smiles": smiles,
            "pKa": pka,
            "source_file": os.path.basename(path)
        })

    return pd.DataFrame(records)


def main():
    os.makedirs("data/processed", exist_ok=True)

    all_files = sorted(
        path
        for path in glob.glob(os.path.join(RAW_DIR, "*.sdf"))
        if not os.path.basename(path).startswith("INCORRECT_")
    )
    print(f"Found {len(all_files)} valid SDF files (INCORRECT_* excluded)")

    dfs = []
    for f in all_files:
        print(f"Parsing {f}")
        df = parse_sdf(f)
        print(f"  -> {len(df)} entries")
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    print(f"Total before cleaning: {len(full_df)}")

    # Clean
    full_df["pKa"] = pd.to_numeric(full_df["pKa"], errors="coerce")
    full_df = full_df.dropna(subset=["pKa"])

    # Remove extreme nonsense values
    full_df = full_df[(full_df["pKa"] > -5) & (full_df["pKa"] < 20)]

    # Deduplicate by canonical smiles + pKa
    full_df = full_df.drop_duplicates(subset=["smiles", "pKa"])

    print(f"Total after cleaning: {len(full_df)}")

    full_df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
