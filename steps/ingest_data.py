import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Data ingestion class which ingests data and returns a DataFrame.
    """

    def __init__(self, data_path: str):
        """Initializing the data ingestion class.
        Args: data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Args: None
    Returns: df: pd.DataFrame
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"serror while ingesting the data: {e}")
        raise e