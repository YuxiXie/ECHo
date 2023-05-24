import requests
from tqdm import tqdm


PREFIX = "http://transcripts.foreverdreaming.org/viewtopic.php?f=34&t="
PAGE_INDEX = {
    'S1': [13183, 13205],
    'S2': [13206, 13235],
    'S3': [13236, 13258],
    'S4': [13259, 13282],
    'S5': [13283, 13309],
    'S6': [13310, 13333],
    'S7': [13334, 13357],
    'S8': [13358, 13374],
    'S9': [13375, 13414],
    'S10': [13415, 13437],
    'S11': [13438, 13459],
    'S12': [13460, 13481],
    'S13': [13482, 13503],
    'S14': [13504, 13525],
}


def get_season_corpus(page_start, page_end, s_id):
    season = {}
    for i, page in tqdm(enumerate(range(page_start, page_end + 1))):
        url = f'{PREFIX}{page}'
        T = get_html(url)
        if T is None: continue

        season['S{:0>2d}E{:0>2d}'.format(s_id, i + 1)] = T
    
    return season


def dump_season_html(season):
    for k, v in season.items():
        with open(f'../html_data/{k}.html', 'w', encoding='utf-8') as f:
            f.write(v.text)


def get_html(url):
    session = requests.Session()
    try:
        resp = session.get(url)
        return resp
    except Exception as e:
        print("get_soup: %s : %s" % (type(e), url))
        return None


if __name__ == '__main__':
    ##=== Download the HTML version of transcripts ===##
    for k, v in PAGE_INDEX.items():
        if len(v) == 2:
            season = get_season_corpus(v[0], v[1], int(k[1:]))
            dump_season_html(season)
