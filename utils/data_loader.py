import pandas as pd

def load_csv_info(df: pd.DataFrame):
    """Kembalikan informasi singkat dataset untuk ditampilkan di UI."""
    info = {
        'rows': len(df),
        'cols': len(df.columns),
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return info