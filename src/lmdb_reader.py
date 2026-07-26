import lmdb
from pathlib import Path
from typing import Union, Optional, Any, Dict
from src.logger import get_logger

logger = get_logger("DocForge.LMDBReader")

class LMDBReader:
    """A safe, memory-efficient reader for LMDB databases.

    Designed for handling large dataset files like DocTamper's .mdb databases.
    Implements the context manager protocol to ensure safe resource disposal.
    """
    _env_cache: Dict[str, lmdb.Environment] = {}
    _ref_counts: Dict[str, int] = {}

    def __init__(
        self,
        db_path: Union[str, Path],
        readonly: bool = True,
        lock: bool = False
    ) -> None:
        """Initialize the LMDBReader.

        Args:
            db_path: Path to the directory containing data.mdb and lock.mdb.
            readonly: Whether to open the database in read-only mode.
            lock: Whether to use database locking.
        """
        self.db_path = Path(db_path).resolve()
        self.readonly = readonly
        self.lock = lock
        self.env: Optional[lmdb.Environment] = None
        self._num_samples: Optional[int] = None

    def open(self) -> "LMDBReader":
        """Open the LMDB database environment if not already opened.

        Returns:
            LMDBReader: Self.
        
        Raises:
            FileNotFoundError: If the database path does not exist.
            lmdb.Error: For database opening errors.
        """
        if self.env is not None:
            return self

        if not self.db_path.exists():
            logger.error(f"Database path does not exist: {self.db_path}")
            raise FileNotFoundError(f"Database path not found: {self.db_path}")

        path_str = str(self.db_path.resolve())
        
        # Check cache
        if path_str in LMDBReader._env_cache:
            self.env = LMDBReader._env_cache[path_str]
            LMDBReader._ref_counts[path_str] += 1
            logger.debug(f"Reused cached database environment for {self.db_path.name}. Ref count: {LMDBReader._ref_counts[path_str]}")
            return self

        logger.info(f"Opening database at {self.db_path.name}...")
        try:
            # Open the environment with parameters tailored for performance and safety
            # readonly=True, lock=False, readahead=False, meminit=False are highly recommended
            # for deep learning read-only database pipelines.
            self.env = lmdb.open(
                str(self.db_path),
                readonly=self.readonly,
                lock=self.lock,
                readahead=False,
                meminit=False
            )
            # Store in cache
            LMDBReader._env_cache[path_str] = self.env
            LMDBReader._ref_counts[path_str] = 1
            logger.info("Database opened successfully.")
        except lmdb.Error as e:
            logger.error(f"Failed to open LMDB database at {self.db_path}: {e}")
            raise

        return self

    def close(self) -> None:
        """Close the LMDB database environment safely."""
        if self.env is not None:
            path_str = str(self.db_path.resolve())
            if path_str in LMDBReader._ref_counts:
                LMDBReader._ref_counts[path_str] -= 1
                ref = LMDBReader._ref_counts[path_str]
                logger.debug(f"Closed reference to database {self.db_path.name}. Ref count remaining: {ref}")
                if ref <= 0:
                    self.env.close()
                    LMDBReader._env_cache.pop(path_str, None)
                    LMDBReader._ref_counts.pop(path_str, None)
                    logger.info(f"Database at {self.db_path.name} fully closed.")
            else:
                self.env.close()
                logger.info(f"Database at {self.db_path.name} closed (uncached).")
            self.env = None

    def __enter__(self) -> "LMDBReader":
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get(self, key: Union[str, bytes]) -> Optional[bytes]:
        """Retrieve the binary value for a given key from the database.

        Args:
            key: The database key (as a string or bytes).

        Returns:
            Optional[bytes]: The byte value if key is found, None otherwise.
        """
        if self.env is None:
            self.open()

        key_bytes = key if isinstance(key, bytes) else key.encode("utf-8")
        
        try:
            # Start a read transaction
            with self.env.begin(write=False) as txn:
                val = txn.get(key_bytes)
                return val
        except lmdb.Error as e:
            logger.error(f"Error reading key {key} from {self.db_path.name}: {e}")
            return None

    def get_num_samples(self) -> int:
        """Retrieve the number of samples in the dataset.

        Checks the metadata key 'num-samples' first, otherwise infers
        it from the number of keys.

        Returns:
            int: The total number of image-label pairs.
        """
        if self._num_samples is not None:
            return self._num_samples

        if self.env is None:
            self.open()

        # Try b'num-samples' metadata key first
        num_samples_bytes = self.get(b"num-samples")
        if num_samples_bytes is not None:
            try:
                self._num_samples = int(num_samples_bytes.decode("utf-8").strip())
                return self._num_samples
            except (ValueError, UnicodeDecodeError):
                logger.warning(
                    f"Found 'num-samples' key but failed to parse value: {num_samples_bytes}. "
                    "Inferred count will be used instead."
                )

        # Fallback to stat entries
        # Since each sample has an image and a label, plus metadata keys,
        # we check the entries count and divide by 2.
        try:
            stat = self.env.stat()
            entries = stat["entries"]
            # Exclude non-sample keys if any
            # We assume total entries include 'num-samples', so let's check
            # if we can do an estimation.
            has_num_samples = num_samples_bytes is not None
            sample_entries = entries - 1 if has_num_samples else entries
            self._num_samples = sample_entries // 2
            return self._num_samples
        except lmdb.Error as e:
            logger.error(f"Failed to read database stats for sample count: {e}")
            return 0
            
    def get_stats(self) -> dict:
        """Get structural stats from LMDB environment.

        Returns:
            dict: LMDB stats dictionary.
        """
        if self.env is None:
            self.open()
        return self.env.stat()
