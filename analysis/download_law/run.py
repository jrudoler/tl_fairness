"""Download the pinned law dataset; see config/datasets.json."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.tools.download_dataset import download

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/raw/law.csv')
    print(download('law',parser.parse_args().output))
