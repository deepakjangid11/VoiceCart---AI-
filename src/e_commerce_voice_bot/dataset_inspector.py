from pathlib import Path
import csv

RAW_DATA_DIR = Path("data/raw")

def inspect_csv(file_path : Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"FILE: {file_path.name}")
    print(f"SIZE: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    with file_path.open("r",encoding = "utf-8-sig",newline = "") as file:
         reader = csv.reader(file)
         headers = next(reader)
         print(f"COLUMNS: {len(headers)}")
         print("HEADERS:")

         for column in headers:
              print(f"  - {column}")

         rows = []

         for index , row in enumerate(reader):
              if index >= 5:
                   break
              rows.append(row)

         print("\nFIRST 5 ROWS:")
         for row in rows:
              print(row)

def main() -> None:
     csv_files = list(RAW_DATA_DIR.glob("*.csv"))

     if not csv_files:
          print("No CSV files found in data/raw/")

     for csv_file in csv_files:
        inspect_csv(csv_file)


if __name__ == "__main__":
    main() 
    