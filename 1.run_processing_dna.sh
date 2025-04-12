
# download here: Homo_sapiens.GRCh38.dna.primary_assembly.fa
## https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

for i in {1..22}
do
    python bin/process_genome.py \
        --genome Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --chroms $i \
        --out test_data 
done