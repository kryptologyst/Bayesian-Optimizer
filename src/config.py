from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "ML Project"
    version: str = "1.0.0"
    random_state: int = 42
    
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    output_dir: Path = base_dir / "outputs"
    plots_dir: Path = output_dir / "plots"
    models_dir: Path = output_dir / "models"
    
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
