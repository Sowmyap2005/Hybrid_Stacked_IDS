# Data availability

## Dataset used

The experiments use the **CIC-IDS2017** dataset from the Canadian Institute for Cybersecurity.

Official source:

https://www.unb.ca/cic/datasets/ids-2017.html

The repository does not redistribute the original CSV files. Users should download the dataset from the official source and follow the dataset provider's applicable terms.

## Expected local layout

Place the downloaded CSV files here:

```text
data/
└── TrafficLabelling/
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    └── Wednesday-workingHours.pcap_ISCX.csv
```

Alternatively, set the `CICIDS2017_DIR` environment variable to the directory containing these files.

## Reproducibility note

The supplied preprocessing samples 20% of each CSV using a fixed seed, cleans the resulting data, removes rare classes, creates a stratified train/test split, reduces the training partition for memory control, standardizes features using training-only statistics, and applies SMOTE only to the training partition.

The raw dataset is therefore an external input to the repository rather than a versioned repository artifact.
