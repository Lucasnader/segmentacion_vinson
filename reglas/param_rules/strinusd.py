# strinusd.py
from stri_stro_common import _run_str, PCTS_DEF
def run_parameters_strinusd(path: str, *, subsubsegments=None, percentiles=PCTS_DEF):
    return _run_str(path, direction="Inbound", currency="USD",
                    subsubsegments=subsubsegments, percentiles=percentiles)
