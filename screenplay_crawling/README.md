# CSI Corpus Pre-Processing

This is the code to download and preprocess the [CSI transcripts](http://transcripts.foreverdreaming.org/viewforum.php?f=34).

## Folder Structure

```
CSI
 ├─ vocab
 └─ raw_data
       └─ html_data
```

## Download

```bash
python crawl_data.py
```
The transcripts will be crawled and saved in [`raw_data/html_data`](./raw_data/html_data) as `.html` file per episode.

## Pre-Processing

Parse the HTML files to get splited blocks of transcripts (10 min)
```bash
python collect_data.py
```
The organized data will be saved as [`raw_data/full_csi_data.json`](./raw_data/full_csi_data.json) (`OUTFILENAME`).

Extract the characters' names for each block of transcripts (3 min)
```bash
python characters_filter.py
```
This will update the data saved in [`raw_data/full_csi_data.json`](./raw_data/full_csi_data.json) (`OUTFILENAME`) by adding the character info.


