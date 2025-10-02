import dbm
import csv
import argparse

def make_crosswalk_dbm(input_csv, output_dbm):
    """
    Create a dbm database from a CSV file containing crosswalk data.
    
    Parameters:
    - input_csv: Path to the input CSV file. The CSV should have two columns:
                 'NWM_ID' and 'NextGen_ID'.
    - output_dbm: Path to the output dbm database file.
    """
    r = 0
    with dbm.open(output_dbm, 'c') as db:
        with open(input_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    feature_id = int(row['NWM_ID'])
                    catchment_number = int(float(row['NextGen_ID'][4:]))
                    db[catchment_number.to_bytes(4, 'big')] = feature_id.to_bytes(8, 'big')
                except Exception as e:
                    print(f"ERROR processing row with values {row['NWM_ID']}, {row['NextGen_ID']}")
                r += 1
                if r%10000 == 0:
                    print(f"Finished {r} rows...")
    print(f"Finished {r} rows.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a crosswalk dbm database from a CSV file.")
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_dbm', type=str, help='Path to the output dbm database file.')

    args = parser.parse_args()
    
    make_crosswalk_dbm(args.input_csv, args.output_dbm)
    
