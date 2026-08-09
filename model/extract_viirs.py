import zipfile
import os

ZIP_DIR = "."

for y in range(2019, 2024):
    zip_path = os.path.join(ZIP_DIR, f"viirs-snpp_{y}_all_countries.zip")
    member = f"viirs-snpp/{y}/viirs-snpp_{y}_Indonesia.csv"
    with zipfile.ZipFile(zip_path) as z:
        z.extract(member, "real_data")
    print("extracted:", member)