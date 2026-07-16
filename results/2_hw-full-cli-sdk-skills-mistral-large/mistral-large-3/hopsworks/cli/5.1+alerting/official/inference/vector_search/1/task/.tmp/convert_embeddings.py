import csv
import json

input_file = "data/items.csv"
output_file = ".tmp/items_processed.csv"

with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        embedding_str = row['embedding']
        embedding_list = json.loads(embedding_str)
        row['embedding'] = ','.join(map(str, embedding_list))
        writer.writerow(row)

print(f"Processed file saved to {output_file}")