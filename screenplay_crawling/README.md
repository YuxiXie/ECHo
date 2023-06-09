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
You can modify `PAGE_INDEX` to specify the seasons to crawl.
For later processing, we only focus on data till Season 8 as the others do not contain scene segmentation (indicated by `<hr />`).

## Pre-Processing

Parse the HTML files to get splited blocks of transcripts (10 min)
```bash
python collect_data.py
```
The organized data will be saved as [`raw_data/full_csi_data.json`](./raw_data/full_csi_data.json) (`OUTFILENAME`).

**Note**: Since the [foreverdreaming website pages](https://transcripts.foreverdreaming.org/viewforum.php?f=34) have been updated, we provide the previous [`html_data`](https://drive.google.com/file/d/1ITFe0n1RjxUNAmDanQJoueul8DUCrnPj/view?usp=sharing) where the current processing code still work.

Extract the characters' names for each block of transcripts (3 min)
```bash
python characters_filter.py
```
This will update the data saved in [`raw_data/full_csi_data.json`](./raw_data/full_csi_data.json) (`OUTFILENAME`) by adding the character info.


