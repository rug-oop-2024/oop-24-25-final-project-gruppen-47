from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """Artifact registry to store and retrieve artifacts"""

    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initialize the artifact registry.

        Args:
            database (Database): Database to store metadata.
            storage (Storage): Storage to store artifacts.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact in the registry.

        Args:
            artifact (Artifact): Artifact to register.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        List all artifacts in the registry.

        Args:
            type (str): Type of artifact to list. Defaults to None.

        Returns:
            List[Artifact]: List of artifacts.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Get an artifact from the registry.

        Args:
            artifact_id (str): Artifact ID.

        Returns:
            Artifact: Artifact object.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact from the registry.

        Args:
            artifact_id (str): Artifact ID.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """AutoML System"""

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the AutoML System.

        Args:
            storage (LocalStorage): Local storage.
            database (Database): Database to store information.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Get the singleton instance of the AutoML System

        Returns:
            AutoMLSystem: The singleton instance of the AutoML System
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Get the artifact registry

        Returns:
            ArtifactRegistry: The artifact registry
        """
        return self._registry
