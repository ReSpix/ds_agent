from datetime import datetime
from pathlib import Path

class ArtifactSaver:
    def __init__(self, base_dir: str, extension: str):
        self.base_dir = Path(base_dir)
        self.extension = extension

    def save(self, type: str, content: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        filename = f"{timestamp}.{self.extension}"

        file_path = self.base_dir / type / filename

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
