# strotclp.py
from stri_stro_common import _run_str, PCTS_DEF
def run_parameters_strotclp(path: str, *, subsubsegments=None, percentiles=PCTS_DEF):
    return _run_str(path, direction="Outbound", currency="CLP",
                    subsubsegments=subsubsegments, percentiles=percentiles)
