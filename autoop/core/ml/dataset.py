from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Dataset"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, 
                       version: str = "1.0.0") -> "Dataset":
        """
        Creates a dataset from a pandas DataFrame

        Args:
            data (pd.DataFrame): DataFrame
            name (str): Name of the dataset
            asset_path (str): Path to the dataset asset
            version (str, optional): Version of the dataset. Defaults to "1.0.0"
        
        Returns:
            Dataset: Dataset artifact
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
 
    def read(self) -> pd.DataFrame:
        """Reads the dataset from the artifact"""
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Saves the dataset to the artifact"""
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
  